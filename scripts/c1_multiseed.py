#!/usr/bin/env python3
"""C1: multi-seed ablation of the final LSTM-MDN training.

For each market, we retrain the best configuration (same BEST_CONFIGS used in
c3_psv_per_fold.py) on the 85/15 chronological split with early stopping, for
multiple random seeds. We record the minimum validation NLL reached per seed
and report mean +/- std across seeds.

This is a cheaper variant of the walk-forward cross-validation (single fold,
final-training regime) intended to characterise seed-induced variability in
the reported validation NLL of Table 4.

Output: data/c1_multiseed.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

TRAIN_FRAC = 0.85
MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-3
BATCH_SIZE = 256
SEEDS = (42, 0, 1, 2)


def load_xi(market: str) -> np.ndarray:
    z = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    return z["xi"].astype(np.float32)


def make_split(xi, lookback):
    T = len(xi)
    n = T - lookback
    X = np.stack([xi[i:i + lookback] for i in range(n)])
    y = xi[lookback:]
    split = int(n * TRAIN_FRAC)
    return X[:split], y[:split], X[split:], y[split:]


def train_once(X_tr, y_tr, X_va, y_va, device, cfg, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

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


def main():
    device = torch.device("cpu")
    out = DATA / "c1_multiseed.txt"

    header = [
        "=" * 90,
        "  C1 -- multi-seed ablation of final-training MDN val NLL (Phase 1, 85/15 split)",
        "=" * 90,
        f"  seeds={SEEDS}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  lr={LR}",
        "",
        f"  {'market':<6}" + "".join([f"  seed={s:<4}" for s in SEEDS])
        + f"  {'mean':>10}  {'std':>8}",
        "  " + "-" * 80,
    ]
    out.write_text("\n".join(header) + "\n", encoding="utf-8")

    rows = {}
    for market, cfg in BEST_CONFIGS.items():
        xi = load_xi(market)
        X_tr, y_tr, X_va, y_va = make_split(xi, cfg["lookback"])
        print(f"[{market}] train={len(y_tr)}  val={len(y_va)}", flush=True)

        vals = []
        for s in SEEDS:
            v = train_once(X_tr, y_tr, X_va, y_va, device, cfg, s)
            vals.append(v)
            print(f"  seed={s}:  val_nll={v:+.4f}", flush=True)
        vals = np.array(vals)
        rows[market] = vals

        with open(out, "a", encoding="utf-8") as fh:
            cells = "".join([f"  {v:>9.4f}" for v in vals])
            fh.write(
                f"  {market:<6}{cells}  {vals.mean():>+10.4f}  {vals.std(ddof=1):>8.4f}\n"
            )

    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\nInterpretation:\n")
        fh.write(
            "  std quantifies seed-induced variability of the Phase 1 validation NLL\n"
            "  on the fixed 85/15 chronological split; complements the across-fold std\n"
            "  reported in Table 4 (walk-forward cross-validation).\n"
        )
        fh.write("=" * 90 + "\n")

    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
