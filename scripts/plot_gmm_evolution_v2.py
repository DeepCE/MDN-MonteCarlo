#!/usr/bin/env python3
"""Fig. 8 rigenerata con selezione sistematica dei timestamp (risposta R1.8.b).

Per ciascun mercato:
  1. Carica modello MDN v2 + detrended xi.
  2. Calcola realized volatility RV_t (rolling 30-day std di r_t = xi_t - xi_{t-1}).
  3. Seleziona 4 timestamp rappresentativi come quantili di RV_t:
     Q10 (calm), Q50 (typical), Q90 (high-vol), Q99 (crisis).
  4. Per ogni timestamp: forward pass MDN su finestra x_t, ottieni GMM (pi,mu,sigma)_t.
  5. La distribuzione del log-return r_{t+1} = xi_{t+1} - xi_t e' ottenuta traslando
     la GMM di -xi_t: r ~ sum_k pi_k N(mu_k - xi_t, sigma_k^2).
  6. Plotta GMM traslata vs Student-t unconditional fittata su tutti i r empirici.

Output:
  figures/gmm_evolution_v2_<market>.png   (ciascuno 4 panel, una riga)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from mdn_models import load_checkpoint_model  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MARKETS = ("psv", "pun", "pjm", "wti")
MARKET_LABELS = {
    "psv": "PSV gas (IT)",
    "pun": "PUN power (IT)",
    "pjm": "PJM power (US)",
    "wti": "WTI crude oil (US)",
}

# Quantili di RV da usare come "regimi"
QUANTILES = (0.10, 0.50, 0.90, 0.99)
REGIME_LABELS = ("Q10 (calm)", "Q50 (typical)", "Q90 (high volatility)", "Q99 (crisis)")

ROLL_WIN = 30


def realized_volatility(r: np.ndarray, window: int = ROLL_WIN) -> np.ndarray:
    """Rolling std over a past window (trailing)."""
    n = len(r)
    rv = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        rv[i] = r[lo:i + 1].std() if i >= 1 else np.nan
    return rv


def pick_timestamps_by_quantile(rv: np.ndarray, valid_lo: int, valid_hi: int,
                                 quantiles=QUANTILES) -> list[int]:
    """Per ogni quantile, trova il timestep con RV_t piu' vicino a quel quantile.

    Restringe la ricerca a [valid_lo, valid_hi] per rispettare lookback e fine serie.
    """
    rv_v = rv[valid_lo:valid_hi].copy()
    # rimuovi NaN
    valid_mask = ~np.isnan(rv_v)
    targets = np.quantile(rv_v[valid_mask], quantiles)
    idxs = []
    for q_val in targets:
        # Cerca il timestep il cui RV e' piu' vicino al target
        diff = np.abs(rv_v - q_val)
        diff[~valid_mask] = np.inf
        local_idx = int(np.argmin(diff))
        idxs.append(valid_lo + local_idx)
    return idxs


def gmm_density_r(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                  xi_t: float, r_grid: np.ndarray) -> np.ndarray:
    """Density of r = xi_{t+1} - xi_t under the GMM for xi_{t+1}.

    p_r(r) = p_xi(r + xi_t) = sum_k pi_k N(r + xi_t; mu_k, sigma_k^2)
           = sum_k pi_k N(r; mu_k - xi_t, sigma_k^2).
    """
    y = r_grid  # shape (G,)
    # shape (G, K)
    z = (y[:, None] - (mu - xi_t)[None, :]) / sigma[None, :]
    gauss = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sigma[None, :])
    return (pi[None, :] * gauss).sum(axis=1)


def run_market(market: str) -> Path:
    # Load empirical xi and r
    zdet = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    xi = zdet["xi"]
    r = zdet["r"]
    T = len(xi)

    # RV over r (length T-1). Use "trailing" version.
    rv = realized_volatility(r, ROLL_WIN)

    # Load model
    device = torch.device("cpu")
    model, ckpt = load_checkpoint_model(MODELS / f"mdn_v2_{market}.pt", device)
    cfg = ckpt["config"]
    m = cfg["lookback"]
    K = cfg["n_components"]

    # Valid indices for input window x_t = xi[t-m+1:t+1] -> predicts xi[t+1]
    # We need t in [m-1, T-2] (so that xi[t+1] exists and x_t has m samples)
    # Map to rv index: rv is indexed on r, r[i] = xi[i+1]-xi[i], so rv[i] uses past up to i
    # We pick rv indices in [m-1, T-2] (1:1 with xi t indices for window completeness)
    valid_lo = m - 1
    valid_hi = T - 2

    idxs = pick_timestamps_by_quantile(rv, valid_lo, valid_hi)
    # idxs are xi indices t such that x_t = xi[t-m+1:t+1] is the input for predicting xi[t+1]

    # Unconditional Student-t fit on all r
    dof, loc, scale = stats.t.fit(r)
    r_std_emp = r.std()

    # Forward passes
    model.eval()
    gmm_params = []
    with torch.no_grad():
        for t in idxs:
            window = xi[t - m + 1:t + 1].astype(np.float32)  # (m,)
            x = torch.from_numpy(window).reshape(1, m, 1)
            pi, mu, sigma = model(x)
            gmm_params.append((pi.numpy().ravel(),
                                mu.numpy().ravel(),
                                sigma.numpy().ravel(),
                                float(xi[t])))

    # Plot range: ~4 std of returns
    bound = 4.0 * r_std_emp
    r_grid = np.linspace(-bound, bound, 1024)
    t_dens = stats.t.pdf(r_grid, dof, loc=loc, scale=scale)

    # Figure: 1 row, 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharex=True)
    for ax, (pi, mu, sigma, xi_t), label, idx, q in \
            zip(axes, gmm_params, REGIME_LABELS, idxs, QUANTILES):
        p_mdn = gmm_density_r(pi, mu, sigma, xi_t, r_grid)
        ax.plot(r_grid, p_mdn, color="C0", lw=2.0,
                label=r"MDN cond. $p(r_{t+1}|\mathbf{x}_t)$")
        ax.plot(r_grid, t_dens, color="C3", lw=1.6, linestyle="--",
                label="Student-$t$ uncond. fit")
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(f"{label}  (t={idx}, RV={rv[idx]:.4f})", fontsize=11)
        ax.set_xlabel(r"$r_{t+1}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()

    out = FIGURES / f"gmm_evolution_v2_{market}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{market}] saved {out.relative_to(ROOT)}  "
          f"idxs={idxs}  RVs={[f'{rv[i]:.4f}' for i in idxs]}")
    return out


def main():
    for market in MARKETS:
        run_market(market)


if __name__ == "__main__":
    main()
