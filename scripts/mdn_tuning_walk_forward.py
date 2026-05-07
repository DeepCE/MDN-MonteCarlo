#!/usr/bin/env python3
"""Walk-forward 3-fold MDN tuning su 4 mercati, grid esteso legacy + enhanced.

Novita' rispetto a mdn_tuning_minio_v2.py:
  * Architettura come iperparametro: {legacy, enhanced}
    - legacy  : LSTM -> fc_hidden(Linear+ReLU+Dropout) -> heads  (paper v1)
    - enhanced: LSTM -> [FC+GELU+Dropout+FC + residual + LayerNorm] x 2 -> pre_mdn -> heads
  * Dropout rivisto: {0.10, 0.15, 0.20}
  * Walk-forward 3-fold invece di split 80/20 single-fold
    - fold 0: train [0 : 0.55T),  val [0.55T : 0.70T)
    - fold 1: train [0 : 0.70T),  val [0.70T : 0.85T)
    - fold 2: train [0 : 0.85T),  val [0.85T : T)
  * Ogni (market, fold, config) e' un trial separato -> ASHA pota aggressivamente
  * Post-process aggrega val_loss per (market, config) sui 3 fold e seleziona best

Grid per mercato (72 config):
    arch x K x hidden_dim x m x dropout = 2 x 3 x 2 x 2 x 3 = 72
Totale trial: 4 markets x 72 configs x 3 folds = 864

Risposta diretta a R1.5 (grid too narrow + no CV).

Esecuzione:
    RAY_ADDRESS='http://<ray-head>:8265' ray job submit \
        --working-dir . -- python code/mdn_tuning_walk_forward.py

Output:
    s3://ray-cluster/results/walk_forward_v2/
        {market}_all_trials.csv     per-trial val_loss (fold + config)
        {market}_best_config.json   best config by mean val_loss over 3 folds
        summary.txt                 tabella riassuntiva
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ray import tune
from ray.tune.schedulers import ASHAScheduler


# ============================================================================
# COSTANTI E PERCORSI
# ============================================================================

COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_BUCKET = "ray-cluster"
MINIO_RESULTS_PREFIX = "results/walk_forward_v2"


# ============================================================================
# DATA LOADING + LOESS DETRENDING (inline, no external deps)
# ============================================================================

def load_prices(market: str) -> np.ndarray:
    fname = COMMODITY_FILES[market]
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    with open(path, "r") as f:
        return np.array([
            float(line.strip().replace(",", "."))
            for line in f if line.strip()
        ])


def loess_detrend(prices: np.ndarray, frac: float = 0.1) -> np.ndarray:
    log_prices = np.log(prices)
    n = len(log_prices)
    window = max(5, int(n * frac) | 1)
    half_w = window // 2
    x = np.arange(-half_w, half_w + 1)
    sigma = half_w / 2
    weights = np.exp(-x ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()
    padded = np.pad(log_prices, half_w, mode="edge")
    trend = np.array([np.sum(padded[i:i + window] * weights) for i in range(n)])
    return log_prices - trend


# ============================================================================
# WALK-FORWARD 3-FOLD SPLIT
# ============================================================================

FOLD_BOUNDARIES = (0.55, 0.70, 0.85, 1.00)   # train_end_i = boundary[i], val_end_i = boundary[i+1]


def build_fold(detrended: np.ndarray, lookback: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward split. fold in {0,1,2}.

    Per ogni sample (X_i, y_i), y_i e' al tempo (i + lookback).
    Train: y al tempo < train_end. Val: train_end <= y < val_end.
    """
    T = len(detrended)
    train_end = int(T * FOLD_BOUNDARIES[fold])
    val_end = int(T * FOLD_BOUNDARIES[fold + 1])

    n_samples = T - lookback
    targets_t = np.arange(n_samples) + lookback     # tempo del target di sample i

    train_mask = targets_t < train_end
    val_mask = (targets_t >= train_end) & (targets_t < val_end)

    X = np.stack([detrended[i:i + lookback] for i in range(n_samples)])
    y = detrended[lookback:]

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


# ============================================================================
# MDN ARCHITETTURE (inline per evitare import da working-dir)
# ============================================================================

class AutoregressiveMDNLegacy(nn.Module):
    """Paper architecture: LSTM + fc_hidden(Linear+ReLU+Dropout) + 3 heads."""

    def __init__(self, lookback=30, hidden_dim=128, n_layers=2, n_components=5, dropout=0.10):
        super().__init__()
        self.lookback = lookback
        self.n_components = n_components
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        h = self.fc_hidden(h)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4
        return pi, mu, sigma


class EnhancedMDN(nn.Module):
    """Paper v1 enhanced: LSTM + 2 residual FC+LayerNorm + pre_mdn + heads."""

    def __init__(self, lookback=30, hidden_dim=96, n_layers=2, n_components=8,
                 n_hidden_layers=2, dropout=0.15):
        super().__init__()
        self.lookback = lookback
        self.n_components = n_components
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_hidden_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                          for _ in range(n_hidden_layers)])
        self.pre_mdn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)
        nn.init.constant_(self.fc_sigma.bias, -1.0)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            h = norm(layer(h) + h)
        h = self.pre_mdn(h)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4
        return pi, mu, sigma


def build_model(cfg: dict) -> nn.Module:
    if cfg["arch"] == "enhanced":
        return EnhancedMDN(
            lookback=cfg["lookback"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=2,
            n_components=cfg["n_components"],
            n_hidden_layers=2,
            dropout=cfg["dropout"],
        )
    return AutoregressiveMDNLegacy(
        lookback=cfg["lookback"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=2,
        n_components=cfg["n_components"],
        dropout=cfg["dropout"],
    )


def mdn_loss(pi, mu, sigma, target):
    target = target.unsqueeze(-1)
    var = sigma ** 2
    log_prob = -0.5 * ((target - mu) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
    log_pi = torch.log(pi + 1e-10)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


# ============================================================================
# TRAINING LOOP (UN TRIAL = UNA COPPIA market + config + fold)
# ============================================================================

def train_trial(config: dict):
    market = config["market"]
    fold = config["fold"]
    prices = load_prices(market)
    detrended = loess_detrend(prices, frac=0.1)

    X_tr, y_tr, X_va, y_va = build_fold(detrended, config["lookback"], fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.FloatTensor(X_tr).unsqueeze(-1).to(device)
    y_tr = torch.FloatTensor(y_tr).to(device)
    X_va = torch.FloatTensor(X_va).unsqueeze(-1).to(device)
    y_va = torch.FloatTensor(y_va).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=config["batch_size"], shuffle=True,
    )

    model = build_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_val = float("inf")
    patience_counter = 0
    patience = config["patience"]

    for epoch in range(config["max_epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mdn_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(loader))

        model.eval()
        with torch.no_grad():
            val_loss = mdn_loss(*model(X_va), y_va).item()

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        tune.report({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "epoch": epoch,
        })

        if patience_counter >= patience:
            break


# ============================================================================
# POST-PROCESS: AGGREGA PER (market, config) MEDIANDO SUI FOLD
# ============================================================================

CONFIG_KEYS = ("arch", "n_components", "hidden_dim", "lookback", "dropout")


def config_signature(row) -> str:
    """String hashable della config (senza market, fold, batch, lr, etc.)."""
    parts = []
    for k in CONFIG_KEYS:
        col = f"config/{k}"
        parts.append(f"{k}={row[col]}")
    return "|".join(parts)


def aggregate_results(results_df):
    """Ritorna dict: market -> DataFrame aggregated by config."""
    import pandas as pd
    out = {}
    for market in COMMODITY_FILES:
        df = results_df[results_df["config/market"] == market].copy()
        if len(df) == 0:
            continue
        df["cfg_sig"] = df.apply(config_signature, axis=1)
        grouped = df.groupby("cfg_sig").agg(
            val_loss_mean=("best_val_loss", "mean"),
            val_loss_std=("best_val_loss", "std"),
            n_folds=("best_val_loss", "count"),
        ).reset_index()

        # Attach config columns (they are constant within a cfg_sig)
        cfg_cols = {k: df.groupby("cfg_sig")[f"config/{k}"].first()
                    for k in CONFIG_KEYS}
        for k, ser in cfg_cols.items():
            grouped[k] = grouped["cfg_sig"].map(ser)

        # Keep only configs with all 3 folds completed (n_folds==3)
        full = grouped[grouped["n_folds"] == 3].copy()
        full = full.sort_values("val_loss_mean").reset_index(drop=True)
        out[market] = full
    return out


def upload_minio(local_path: Path, remote_key: str):
    try:
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem(endpoint_override=MINIO_ENDPOINT)
        remote_full = f"{MINIO_BUCKET}/{remote_key}"
        with open(local_path, "rb") as src, fs.open_output_stream(remote_full) as dst:
            dst.write(src.read())
        print(f"  [minio] {local_path.name} -> s3://{remote_full}")
    except Exception as e:
        print(f"  [minio] WARN upload failed ({type(e).__name__}): {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  MDN WALK-FORWARD TUNING  (4 markets x 72 configs x 3 folds = 864 trials)")
    print("=" * 80)

    n_configs_per_market = 2 * 3 * 2 * 2 * 3   # arch x K x h x m x dropout
    n_trials_per_market = n_configs_per_market * 3
    print(f"  Per market: {n_configs_per_market} configs x 3 folds = {n_trials_per_market} trials")
    print(f"  Total: {4 * n_trials_per_market} trials")
    print(f"  ASHA: reduction_factor=3, grace_period=30, max_t=200")
    print(f"  Walk-forward boundaries: {FOLD_BOUNDARIES}")

    search_space = {
        "market":       tune.grid_search(list(COMMODITY_FILES.keys())),  # 4
        "fold":         tune.grid_search([0, 1, 2]),                      # 3
        "arch":         tune.grid_search(["legacy", "enhanced"]),         # 2
        "n_components": tune.grid_search([5, 8, 10]),                     # 3
        "hidden_dim":   tune.grid_search([64, 128]),                      # 2
        "lookback":     tune.grid_search([20, 30]),                       # 2
        "dropout":      tune.grid_search([0.10, 0.15, 0.20]),             # 3
        "lr":           1e-3,
        "batch_size":   256,
        "max_epochs":   200,
        "patience":     30,
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=200,
        grace_period=30,
        reduction_factor=3,
    )

    analysis = tune.run(
        train_trial,
        config=search_space,
        resources_per_trial={"gpu": 0.5, "cpu": 2},
        num_samples=1,
        scheduler=scheduler,
        storage_path="/tmp/ray_results",
        name="mdn-walk-forward-v2",
        verbose=1,
        max_concurrent_trials=6,   # 3 GPU x 2 trials/GPU (share 0.5)
    )

    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS BY (market, config) MEANING OVER 3 FOLDS")
    print("=" * 80)

    results_df = analysis.results_df.copy()
    aggregated = aggregate_results(results_df)

    out_dir = Path("/tmp/walk_forward_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = ["=" * 80,
                     "  MDN walk-forward 3-fold tuning — best config per market",
                     "=" * 80]

    for market, df in aggregated.items():
        n_completed = len(df)
        csv_path = out_dir / f"{market}_all_trials.csv"
        df.to_csv(csv_path, index=False)
        upload_minio(csv_path, f"{MINIO_RESULTS_PREFIX}/{market}_all_trials.csv")

        if n_completed == 0:
            summary_lines.append(f"\n-- {market.upper()}: no full-3-fold configs completed")
            continue

        best = df.iloc[0]
        best_cfg = {k: (int(best[k]) if k in ("n_components","hidden_dim","lookback") else
                       (float(best[k]) if k == "dropout" else str(best[k])))
                    for k in CONFIG_KEYS}
        best_cfg["val_loss_mean"] = float(best["val_loss_mean"])
        best_cfg["val_loss_std"] = float(best["val_loss_std"])
        best_cfg["n_folds"] = int(best["n_folds"])
        best_cfg["n_configs_completed"] = n_completed

        json_path = out_dir / f"{market}_best_config.json"
        with open(json_path, "w") as f:
            json.dump(best_cfg, f, indent=2)
        upload_minio(json_path, f"{MINIO_RESULTS_PREFIX}/{market}_best_config.json")

        summary_lines.append(
            f"\n-- {market.upper()}  ({n_completed} full-3-fold configs)"
            f"\n  best: arch={best_cfg['arch']}  K={best_cfg['n_components']}  "
            f"h={best_cfg['hidden_dim']}  m={best_cfg['lookback']}  "
            f"dropout={best_cfg['dropout']}"
            f"\n  val NLL mean +/- std  over 3 folds: "
            f"{best_cfg['val_loss_mean']:+.4f} +/- {best_cfg['val_loss_std']:.4f}"
        )
        # Mostra anche top-5 per avere contesto
        summary_lines.append(f"  top-5 by mean val NLL:")
        for i in range(min(5, len(df))):
            r = df.iloc[i]
            summary_lines.append(
                f"    [{i+1}] arch={r['arch']:<9} K={int(r['n_components'])} "
                f"h={int(r['hidden_dim'])} m={int(r['lookback'])} "
                f"d={r['dropout']:.2f}  "
                f"mean={r['val_loss_mean']:+.4f} std={r['val_loss_std']:.4f}"
            )

    summary_lines.append("\n" + "=" * 80)
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    upload_minio(summary_path, f"{MINIO_RESULTS_PREFIX}/summary.txt")

    print(f"\n  all artifacts saved to:")
    print(f"    local  : {out_dir}/")
    print(f"    minio  : s3://{MINIO_BUCKET}/{MINIO_RESULTS_PREFIX}/")
    print("=" * 80)
