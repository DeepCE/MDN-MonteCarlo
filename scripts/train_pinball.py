#!/usr/bin/env python3
"""Retraining MDN with combined NLL + pinball loss at regulatory quantiles.

Rationale (Framework Modularity, Section 5): the LSTM-MDN framework is
indifferent to the choice of training objective. NLL targets density fit and
is appropriate for moment-level validation. For VaR calibration at regulatory
alpha levels, the pinball loss on conditional quantiles is the strictly
consistent objective (Koenker-Bassett 1978; Gneiting 2011).

We retrain PJM and WTI (markets with most VaR rejections under NLL training)
with:

    L_total = NLL  +  lambda * sum_{alpha} pinball_alpha(r, q_hat_alpha(GMM(x_t)))

with alpha in {0.01, 0.05, 0.95, 0.99}.  The quantile q_hat_alpha is extracted
from the GMM CDF via differentiable grid-based linear interpolation.

Output:
  models/mdn_v2_{market}_pinball.pt
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
MODELS = ROOT / "models"

# Same BEST_CONFIGS as walk-forward selection
BEST_CONFIGS = {
    "pjm": dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
                n_components=5, dropout=0.15, n_hidden_layers=2),
    "wti": dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
                n_components=5, dropout=0.15, n_hidden_layers=2),
}

ALPHAS = (0.01, 0.05, 0.95, 0.99)  # overridable via CLI
LAMBDA = 0.5
TRAIN_FRAC = 0.85
MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-3
BATCH_SIZE = 256
SEED = 42
GRID_SIZE = 512


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


def gmm_quantile_diff(pi, mu, sigma, alpha, n_grid=GRID_SIZE):
    """Differentiable GMM quantile via grid-based CDF inversion.

    pi, mu, sigma: [B, K]  alpha: scalar in (0,1)
    Returns q of shape [B] with dq/d(pi,mu,sigma) flowing.
    """
    B, K = pi.shape
    lo = (mu - 6 * sigma).min(dim=1, keepdim=True).values  # [B,1]
    hi = (mu + 6 * sigma).max(dim=1, keepdim=True).values  # [B,1]
    t = torch.linspace(0, 1, n_grid, device=pi.device).unsqueeze(0)  # [1,G]
    grid = lo + (hi - lo) * t  # [B,G]

    z = (grid.unsqueeze(1) - mu.unsqueeze(2)) / sigma.unsqueeze(2)  # [B,K,G]
    cdf_k = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    cdf = (pi.unsqueeze(2) * cdf_k).sum(dim=1)  # [B,G]

    # For each row find first grid index where cdf >= alpha
    target = torch.full((B, 1), alpha, device=pi.device)
    idx = torch.searchsorted(cdf, target).clamp(1, n_grid - 1).squeeze(1)  # [B]

    i0 = (idx - 1).unsqueeze(1)
    i1 = idx.unsqueeze(1)
    y0 = cdf.gather(1, i0).squeeze(1)
    y1 = cdf.gather(1, i1).squeeze(1)
    x0 = grid.gather(1, i0).squeeze(1)
    x1 = grid.gather(1, i1).squeeze(1)
    q = x0 + (alpha - y0) / (y1 - y0 + 1e-12) * (x1 - x0)
    return q


def pinball(r, q, alpha):
    u = r - q
    return torch.where(u >= 0, alpha * u, (alpha - 1) * u).mean()


def combined_loss(pi, mu, sigma, target, lam=LAMBDA, alphas=None):
    if alphas is None:
        alphas = ALPHAS
    nll = mdn_loss(pi, mu, sigma, target)
    pin = torch.zeros((), device=target.device)
    for a in alphas:
        q = gmm_quantile_diff(pi, mu, sigma, a)
        pin = pin + pinball(target, q, a)
    return nll + lam * pin, nll.detach(), pin.detach()


def train_one_market(market, cfg, device, alphas=None, ckpt_suffix="_pinball"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)

    xi = load_xi(market)
    X_tr, y_tr, X_va, y_va = make_split(xi, cfg["lookback"])
    print(f"[{market}] train={len(y_tr)}  val={len(y_va)}", flush=True)

    Xt = torch.from_numpy(X_tr).unsqueeze(-1).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_va).unsqueeze(-1).to(device)
    yv = torch.from_numpy(y_va).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(cfg, arch=cfg["arch"]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    best_state = None
    counter = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss, _, _ = combined_loss(*model(xb), yb, alphas=alphas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            loss_v, nll_v, pin_v = combined_loss(*model(Xv), yv, alphas=alphas)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep {epoch+1:3d}  val L={loss_v.item():+.4f} "
                  f"(NLL={nll_v.item():+.4f} pin={pin_v.item():+.4f})",
                  flush=True)
        if loss_v.item() < best:
            best = loss_v.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        if counter >= PATIENCE:
            print(f"  early stopped at epoch {epoch+1}  best val L={best:+.4f}",
                  flush=True)
            break

    model.load_state_dict(best_state)
    return model, best, xi


def save_checkpoint(model, cfg, market, best_val, xi, ckpt_suffix="_pinball"):
    MODELS.mkdir(parents=True, exist_ok=True)
    out_path = MODELS / f"mdn_v2_{market}{ckpt_suffix}.pt"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {**cfg},
        "market": market,
        "phase1_best_val": float(best_val),
        "initial_history": xi[-cfg["lookback"]:].astype(np.float32),
        "lookback": cfg["lookback"],
        "training_loss": "NLL + lambda*sum_alpha pinball_alpha",
        "lambda": LAMBDA,
        "alphas": list(ALPHAS),
    }
    torch.save(ckpt, out_path)
    print(f"  saved: {out_path.name}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,wti")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS),
                    help="comma-separated list of alpha levels for pinball loss")
    ap.add_argument("--ckpt-suffix", default="_pinball")
    args = ap.parse_args()

    alphas = tuple(float(a) for a in args.alphas.split(","))
    device = torch.device("cpu")
    print(f"device: {device}  seed: {SEED}  lambda: {LAMBDA}  alphas: {alphas}  "
          f"suffix: {args.ckpt_suffix}", flush=True)

    for market in args.markets.split(","):
        if market not in BEST_CONFIGS:
            print(f"[{market}] SKIP: no BEST_CONFIG entry")
            continue
        cfg = BEST_CONFIGS[market]
        print(f"\n{'='*60}\n  PINBALL TRAINING -- {market.upper()} "
              f"(alphas={alphas})\n{'='*60}", flush=True)
        model, best, xi = train_one_market(market, cfg, device, alphas=alphas,
                                            ckpt_suffix=args.ckpt_suffix)
        save_checkpoint(model, cfg, market, best, xi,
                        ckpt_suffix=args.ckpt_suffix)
    print("\n[done]")


if __name__ == "__main__":
    main()
