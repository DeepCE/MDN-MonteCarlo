#!/usr/bin/env python3
"""Training finale MDN v2 per i 4 mercati con best config da walk-forward tuning.

Pipeline two-phase (coerente con paper v1 `mdn_fulldata_*.pt`):
  Phase 1: split 85/15 su xi chronologico -> early stopping su val (patience=25) ->
           registra best_epoch
  Phase 2: re-train da zero su full data (100%) per best_epoch epoche

Esecuzione tipica sul cluster (con GPU):
    RAY_ADDRESS='http://<ray-head>:8265' ray job submit \
        --working-dir . --entrypoint-num-gpus 1 \
        -- python code/train_final_v2.py

Oppure locale (CPU):
    AWS_ACCESS_KEY_ID=<your-access-key> \
    AWS_SECRET_ACCESS_KEY=<your-secret-key> \
    python code/train_final_v2.py

Input:
    MinIO: s3://ray-cluster/results/walk_forward_v2/{market}_best_config.json
    (fallback locale: /tmp/walk_forward_results/{market}_best_config.json)

Output:
    models/mdn_v2_{market}.pt          local
    s3://ray-cluster/models/mdn_v2_{market}.pt
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "code"
DATA = ROOT / "data"
MODELS = ROOT / "models"

sys.path.insert(0, str(SCRIPTS))
from mdn_models import build_model, mdn_loss   # noqa: E402


COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_BUCKET = "ray-cluster"
MINIO_RESULTS_PREFIX = "results/walk_forward_v2"
MINIO_MODELS_PREFIX = "models"

PHASE1_TRAIN_FRAC = 0.85         # split per early stopping phase 1
PHASE1_MAX_EPOCHS = 250
PHASE1_PATIENCE = 25
PHASE1_LR = 1e-3
PHASE1_BATCH = 256

PHASE2_TRAIN_FRAC = 0.95         # split ristretto per phase 2 (95% train + 5% val finale)
PHASE2_MAX_EPOCHS = 300          # cap assoluto
PHASE2_PATIENCE = 20             # early stopping su mini-holdout
MASTER_SEED = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prices(market: str) -> np.ndarray:
    fname = COMMODITY_FILES[market]
    path = DATA / fname
    if not path.exists():
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


def make_sequences(xi: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(xi) - lookback
    X = np.stack([xi[i:i + lookback] for i in range(n)])
    y = xi[lookback:]
    return X, y


# ============================================================================
# MINIO I/O
# ============================================================================

def _get_fs():
    import pyarrow.fs as pafs
    return pafs.S3FileSystem(endpoint_override=MINIO_ENDPOINT)


def load_best_config(market: str) -> dict:
    """Download best_config.json da MinIO; fallback a /tmp/walk_forward_results/."""
    remote_key = f"{MINIO_BUCKET}/{MINIO_RESULTS_PREFIX}/{market}_best_config.json"
    try:
        fs = _get_fs()
        with fs.open_input_stream(remote_key) as f:
            return json.loads(f.read().decode("utf-8"))
    except Exception as e:
        print(f"  [minio] cannot read {remote_key}: {type(e).__name__}")
        # Fallback locale
        local = Path("/tmp/walk_forward_results") / f"{market}_best_config.json"
        if local.exists():
            print(f"  [fallback] loading {local}")
            return json.loads(local.read_text())
        raise RuntimeError(f"best config not found for {market} (MinIO and /tmp)")


def upload_minio(local_path: Path, remote_key: str) -> None:
    try:
        fs = _get_fs()
        with open(local_path, "rb") as src, \
             fs.open_output_stream(f"{MINIO_BUCKET}/{remote_key}") as dst:
            dst.write(src.read())
        print(f"  [minio] uploaded -> s3://{MINIO_BUCKET}/{remote_key}")
    except Exception as e:
        print(f"  [minio] WARN upload failed ({type(e).__name__}): {e}")


# ============================================================================
# TRAINING
# ============================================================================

def train_loop(model, X_train, y_train, X_val, y_val, n_epochs, patience,
               lr, batch_size, device, verbose_every=10):
    """Training loop standard. Return (best_val, best_epoch, history)."""
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    counter = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        tl = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mdn_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(1, len(loader))

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vl = mdn_loss(*model(X_val), y_val).item()
        else:
            vl = float("nan")

        history.append((epoch, tl, vl))

        if X_val is not None and vl < best_val:
            best_val = vl
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % verbose_every == 0 or epoch == n_epochs - 1:
            if X_val is not None:
                print(f"    epoch {epoch+1:3d}  train={tl:+.4f}  val={vl:+.4f}  best@{best_epoch+1}")
            else:
                print(f"    epoch {epoch+1:3d}  train={tl:+.4f}")

        if X_val is not None and counter >= patience:
            print(f"    early stop at epoch {epoch+1} (best @ {best_epoch+1}, val={best_val:+.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_epoch, history


def train_one_market(market: str, device: torch.device, ckpt_suffix: str = "") -> None:
    print(f"\n{'=' * 80}\n  TRAINING v2 — {market.upper()}\n{'=' * 80}")

    # Fix seed per-market (determinism pytorch + cuda)
    seed = MASTER_SEED + abs(hash(market)) % 10_000
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  seed: {seed}")

    cfg = load_best_config(market)
    print(f"  best config: arch={cfg['arch']}  K={cfg['n_components']}  "
          f"h={cfg['hidden_dim']}  m={cfg['lookback']}  dropout={cfg['dropout']}")
    print(f"  walk-forward val_loss: {cfg.get('val_loss_mean', 'n/a'):+.4f} "
          f"+/- {cfg.get('val_loss_std', 0):.4f}")

    # Build model config dict acceptable da mdn_models.build_model
    model_cfg = {
        "lookback": int(cfg["lookback"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "n_layers": 2,
        "n_components": int(cfg["n_components"]),
        "dropout": float(cfg["dropout"]),
    }
    if cfg["arch"] == "enhanced":
        model_cfg["n_hidden_layers"] = 2

    # Data
    prices = load_prices(market)
    xi = loess_detrend(prices, frac=0.1)
    X, y = make_sequences(xi, model_cfg["lookback"])

    T_seq = len(X)
    split = int(T_seq * PHASE1_TRAIN_FRAC)
    X_train = torch.FloatTensor(X[:split]).unsqueeze(-1).to(device)
    y_train = torch.FloatTensor(y[:split]).to(device)
    X_val = torch.FloatTensor(X[split:]).unsqueeze(-1).to(device)
    y_val = torch.FloatTensor(y[split:]).to(device)

    print(f"  data: T_prices={len(prices)}  T_sequences={T_seq}  "
          f"train={split}  val={T_seq - split}")

    # ------------------ PHASE 1: early stopping su val ------------------
    print(f"\n  PHASE 1: 85/15 split, early stopping (patience={PHASE1_PATIENCE})")
    model = build_model(model_cfg, arch=cfg["arch"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    model: {cfg['arch']} MDN, {n_params} params")

    t0 = time.time()
    best_val, best_epoch, hist1 = train_loop(
        model, X_train, y_train, X_val, y_val,
        n_epochs=PHASE1_MAX_EPOCHS, patience=PHASE1_PATIENCE,
        lr=PHASE1_LR, batch_size=PHASE1_BATCH, device=device,
    )
    t1 = time.time() - t0
    print(f"    phase 1 done in {t1:.1f}s  best_val={best_val:+.4f}  best_epoch={best_epoch+1}")

    # ------------------ PHASE 2: retrain su 95% + early stopping su 5% finale ------
    # Mini-holdout (ultimo 5%) per early stopping. Il mini-val e' piu' recente del
    # phase 1 val (che era 15% finali includendo il mini-val + altro), quindi
    # sonda la capacita' del modello sulla coda piu' recente del dataset.
    print(f"\n  PHASE 2: retrain on {PHASE2_TRAIN_FRAC*100:.0f}% + early stopping on "
          f"{(1-PHASE2_TRAIN_FRAC)*100:.0f}% final holdout (patience={PHASE2_PATIENCE})")
    split2 = int(T_seq * PHASE2_TRAIN_FRAC)
    X_p2_tr = torch.FloatTensor(X[:split2]).unsqueeze(-1).to(device)
    y_p2_tr = torch.FloatTensor(y[:split2]).to(device)
    X_p2_va = torch.FloatTensor(X[split2:]).unsqueeze(-1).to(device)
    y_p2_va = torch.FloatTensor(y[split2:]).to(device)
    print(f"    phase 2 split: train={split2}  mini-val={T_seq - split2}")

    model_full = build_model(model_cfg, arch=cfg["arch"]).to(device)
    t2 = time.time()
    best_val2, best_epoch2, hist2 = train_loop(
        model_full, X_p2_tr, y_p2_tr, X_p2_va, y_p2_va,
        n_epochs=PHASE2_MAX_EPOCHS, patience=PHASE2_PATIENCE,
        lr=PHASE1_LR, batch_size=PHASE1_BATCH, device=device,
    )
    # Safety floor: se phase 2 ES si ferma prima di phase 1 best_epoch,
    # re-train da zero fino a phase1_best_epoch (senza ES). Protegge contro
    # stopping prematuro causato da distribution shift nel mini-val 5% finale.
    if (best_epoch2 + 1) < (best_epoch + 1):
        floor = best_epoch + 1
        print(f"    [safety floor] phase 2 ES stopped at {best_epoch2 + 1} < "
              f"phase 1 best_epoch {floor}; retraining for {floor} epochs (no ES)")
        model_full = build_model(model_cfg, arch=cfg["arch"]).to(device)
        _, _, hist2_floor = train_loop(
            model_full, X_p2_tr, y_p2_tr, None, None,
            n_epochs=floor, patience=10**9,
            lr=PHASE1_LR, batch_size=PHASE1_BATCH, device=device,
        )
        hist2 = hist2 + [("floor",) + h[1:] for h in hist2_floor]
    t3 = time.time() - t2
    print(f"    phase 2 done in {t3:.1f}s  best_val={best_val2:+.4f}  "
          f"best_epoch={best_epoch2 + 1}")

    # ------------------ SAVE ------------------
    MODELS.mkdir(parents=True, exist_ok=True)
    out_path = MODELS / f"mdn_v2_{market}{ckpt_suffix}.pt"
    checkpoint = {
        "model_state_dict": model_full.state_dict(),
        "config": {**model_cfg, "arch": cfg["arch"]},
        "market": market,
        "phase1_best_val": float(best_val),
        "phase1_best_epoch": int(best_epoch),
        "phase1_time_s": float(t1),
        "phase2_time_s": float(t3),
        "phase1_history": hist1,
        "phase2_history": hist2,
        "walk_forward_val_loss_mean": cfg.get("val_loss_mean"),
        "walk_forward_val_loss_std": cfg.get("val_loss_std"),
        "n_params": int(n_params),
        "initial_history": xi[-model_cfg["lookback"]:].astype(np.float32),
        "lookback": model_cfg["lookback"],
    }
    torch.save(checkpoint, out_path)
    print(f"  saved: {out_path.relative_to(ROOT)}  ({out_path.stat().st_size / 1024:.1f} KB)")

    upload_minio(out_path, f"{MINIO_MODELS_PREFIX}/{out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    global MASTER_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--ckpt-suffix", default="",
                    help="suffix to append to ckpt filename (e.g. '_seed123')")
    args = ap.parse_args()
    MASTER_SEED = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  seed: {MASTER_SEED}  ckpt_suffix: '{args.ckpt_suffix}'")

    for market in args.markets.split(","):
        try:
            train_one_market(market, device, ckpt_suffix=args.ckpt_suffix)
        except Exception as e:
            print(f"\n[ERROR] {market}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}\n  DONE — v2 models trained (seed={MASTER_SEED})\n{'=' * 80}")


if __name__ == "__main__":
    main()
