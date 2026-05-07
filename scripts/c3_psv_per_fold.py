#!/usr/bin/env python3
"""C3: per-fold val NLL analysis of PSV best config.

Best config (from walk-forward tuning): enhanced, K=5, d=128, m=20, dropout=0.20.
Walk-forward 3-fold expanding window with boundaries (0.55, 0.70, 0.85, 1.00).

For each fold k in {0,1,2}, we retrain the same best config and record
the minimum validation NLL reached. We then report the three per-fold NLLs
alongside the Gaussian baseline NLL for reference.

Output: data/c3_psv_per_fold.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from mdn_models import build_model, mdn_loss  # noqa: E402

DATA = ROOT / "data"

BEST_CONFIGS = {
    "psv": dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
                n_components=5, dropout=0.20, n_hidden_layers=2),
    "pun": dict(arch="enhanced", lookback=30, hidden_dim=128, n_layers=2,
                n_components=8, dropout=0.10, n_hidden_layers=2),
    "pjm": dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
                n_components=5, dropout=0.15, n_hidden_layers=2),
    "wti": dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
                n_components=5, dropout=0.15, n_hidden_layers=2),
}
FOLD_BOUNDARIES = (0.55, 0.70, 0.85, 1.00)
MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-3
BATCH_SIZE = 256
SEED = 42


def load_xi(market: str) -> np.ndarray:
    z = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    return z["xi"].astype(np.float32)


def build_fold(xi: np.ndarray, lookback: int, fold: int):
    T = len(xi)
    train_end = int(T * FOLD_BOUNDARIES[fold])
    val_end = int(T * FOLD_BOUNDARIES[fold + 1])
    n = T - lookback
    targets_t = np.arange(n) + lookback
    train_mask = targets_t < train_end
    val_mask = (targets_t >= train_end) & (targets_t < val_end)
    X = np.stack([xi[i:i + lookback] for i in range(n)])
    y = xi[lookback:]
    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def train_fold(X_tr, y_tr, X_va, y_va, device, cfg):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    Xt = torch.from_numpy(X_tr).unsqueeze(-1).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_va).unsqueeze(-1).to(device)
    yv = torch.from_numpy(y_va).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(cfg, arch=cfg["arch"]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    counter = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = mdn_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = mdn_loss(*model(Xv), yv).item()
        if vl < best:
            best = vl
            counter = 0
        else:
            counter += 1
        if counter >= PATIENCE:
            break
    return best


def run_market_all_folds(market: str, device: torch.device):
    cfg = BEST_CONFIGS[market]
    xi = load_xi(market)
    T = len(xi)

    fold_mdn = []
    fold_gauss = []
    for k in range(3):
        X_tr, y_tr, X_va, y_va = build_fold(xi, cfg["lookback"], k)
        print(f"[{market}] fold {k}: train={len(y_tr)}  val={len(y_va)} ...", flush=True)
        best_val = train_fold(X_tr, y_tr, X_va, y_va, device, cfg)

        train_end = int(T * FOLD_BOUNDARIES[k])
        val_end = int(T * FOLD_BOUNDARIES[k + 1])
        r_train = np.diff(xi[:train_end])
        r_val = np.diff(xi[train_end - 1:val_end])
        mu_tr, sig_tr = r_train.mean(), r_train.std(ddof=1)
        gauss_nll = 0.5 * np.log(2 * np.pi * sig_tr ** 2) + \
                    0.5 * ((r_val - mu_tr) ** 2).mean() / (sig_tr ** 2)

        fold_mdn.append(best_val)
        fold_gauss.append(gauss_nll)
        print(f"  MDN={best_val:+.4f}  Gauss_OOF={gauss_nll:+.4f}  "
              f"delta={best_val - gauss_nll:+.4f}")
    return np.array(fold_mdn), np.array(fold_gauss)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(BEST_CONFIGS.keys()))
    ap.add_argument("--append", action="store_true",
                    help="Append to existing output file (no header)")
    args = ap.parse_args()
    markets_to_run = args.markets.split(",")

    device = torch.device("cpu")
    out = DATA / "c3_oof_gauss_comparison.txt"

    if not args.append:
        header = []
        header.append("=" * 108)
        header.append("  C3 -- per-fold walk-forward val NLL: MDN vs Gaussian baseline (both out-of-fold)")
        header.append("=" * 108)
        header.append(
            f"  {'market':<6} {'fold':<5} {'train_end':<10} {'MDN val NLL':>13} "
            f"{'Gauss OOF NLL':>15} {'MDN - Gauss':>13}"
        )
        header.append("  " + "-" * 95)
        out.write_text("\n".join(header) + "\n", encoding="utf-8")

    results = {}
    for market in markets_to_run:
        mdn, gauss = run_market_all_folds(market, device)
        results[market] = (mdn, gauss)
        with open(out, "a", encoding="utf-8") as fh:
            for k in range(3):
                fh.write(
                    f"  {market:<6} {k:<5} {FOLD_BOUNDARIES[k]:<10.2f} "
                    f"{mdn[k]:>13.4f} {gauss[k]:>15.4f} {mdn[k] - gauss[k]:>+13.4f}\n"
                )
            fh.write(
                f"  {market:<6} mean  {'-':<10} "
                f"{mdn.mean():>13.4f} {gauss.mean():>15.4f} {(mdn - gauss).mean():>+13.4f}\n"
            )
            fh.write("\n")

    if not args.append:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("Interpretation:\n")
            fh.write("  Both MDN and Gaussian are now evaluated out-of-fold: fit on the expanding\n")
            fh.write("  train window, evaluated on the immediately subsequent val window.\n")
            fh.write("  Negative 'MDN - Gauss' means MDN outperforms the Gaussian baseline.\n")
            fh.write("=" * 108 + "\n")

    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
