#!/usr/bin/env python3
"""Genera figure di preprocessing compatte per i 4 mercati.

Layout: 1 riga x 3 colonne per mercato.
  Pannello 1: log-prices con LOESS trend sovrapposto
  Pannello 2: log-returns r_t time series
  Pannello 3: distribuzione empirica di r_t con Gaussian overlay

Output: figures/preprocessing_v2_<market>.png (e copiato in root per LaTeX).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MARKETS = ("psv", "pun", "pjm", "wti")
MARKET_LABELS = {
    "psv": "PSV gas (IT)",
    "pun": "PUN power (IT)",
    "pjm": "PJM power (US)",
    "wti": "WTI crude oil (US)",
}


def plot_market(market: str) -> Path:
    z = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    log_p = z["log_prices"]
    trend = z["trend"]
    r = z["r"]
    n = len(log_p)
    t_price = np.arange(n)
    t_ret = np.arange(1, n)

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))

    # Panel 1: log-prices + LOESS trend
    ax = axes[0]
    ax.plot(t_price, log_p, color="black", lw=0.6, label=r"$\log P_t$")
    ax.plot(t_price, trend, color="C3", lw=1.6, linestyle="--",
            label=r"LOESS trend ($\mathrm{frac}=0.1$)")
    ax.set_xlabel("$t$ (days)", fontsize=9)
    ax.set_ylabel(r"$\log P_t$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(alpha=0.3)

    # Panel 2: r_t time series
    ax = axes[1]
    ax.plot(t_ret, r, color="C0", lw=0.5)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("$t$ (days)", fontsize=9)
    ax.set_ylabel(r"$r_t = \xi_t - \xi_{t-1}$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: distribution of r_t with Gaussian overlay (linear scale)
    ax = axes[2]
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    # Use 4-sigma range to focus on bulk; tails are summarized by the kurt annotation
    rng = 4.0 * sigma
    bins = np.linspace(-rng, rng, 60)
    r_clipped = r[(r >= -rng) & (r <= rng)]
    ax.hist(r_clipped, bins=bins, density=True, color="C0",
            alpha=0.55, edgecolor="none")
    xs = np.linspace(-rng, rng, 400)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="C3", lw=1.4,
            linestyle="--", label="Gaussian fit")
    # Kurtosis / skewness annotation
    kurt = stats.kurtosis(r, fisher=False)
    skew = stats.skew(r)
    ax.text(0.02, 0.96, f"kurt $={kurt:.2f}$\nskew $={skew:+.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.set_xlabel(r"$r_t$", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out = FIGURES / f"preprocessing_v2_{market}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Copy also to paper root
    root_copy = ROOT / f"preprocessing_v2_{market}.png"
    root_copy.write_bytes(out.read_bytes())
    print(f"[{market}] saved {out.name} + {root_copy.name}")
    return out


def main():
    for m in MARKETS:
        plot_market(m)


if __name__ == "__main__":
    main()
