#!/usr/bin/env python3
"""Genera figure compatte di traiettorie (multiple + peak-matched) per i 4 mercati.

Due figure per mercato:
  1. multiple: observed + 3 simulated trajectories (typical / calm / volatile)
  2. peak-matched: observed + 1 simulated trajectory that best matches extrema

Dati:
  data/detrended/<market>_detrended.npz          -> empirical xi
  data/mdn_paths/<market>_mdn_paths_v2.npz       -> 5000 MC paths

Output:
  figures/trajectories_v2_<market>.png
  figures/peak_matched_v2_<market>.png
  (copia in root per LaTeX)
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


def select_three_simulated(paths: np.ndarray, emp_xi: np.ndarray,
                            ) -> tuple[int, int, int]:
    """Select three representative simulated trajectories.

    To avoid outliers that dominate the plot's y-axis (e.g. very high-kurt
    WTI paths), we pick via quantiles of the per-path std:
      - typical: matches empirical moments best (min composite score)
      - low-vol : path at ~20th percentile of std (not trivially zero)
      - high-vol: path at ~85th percentile of std (non-outlier)
    """
    r_paths = np.diff(paths, axis=1)
    std_paths = r_paths.std(axis=1)

    r_emp = np.diff(emp_xi)
    emp_std = r_emp.std()
    emp_sk = stats.skew(r_emp)
    emp_ku = stats.kurtosis(r_emp, fisher=False)
    sk_paths = stats.skew(r_paths, axis=1)
    ku_paths = stats.kurtosis(r_paths, axis=1, fisher=False)
    score = (np.abs(std_paths - emp_std) / emp_std
             + np.abs(sk_paths - emp_sk) / (abs(emp_sk) + 0.1)
             + np.abs(ku_paths - emp_ku) / emp_ku)
    idx_typical = int(np.argmin(score))

    q_low = np.quantile(std_paths, 0.20)
    q_high = np.quantile(std_paths, 0.85)
    idx_low = int(np.argmin(np.abs(std_paths - q_low)))
    idx_high = int(np.argmin(np.abs(std_paths - q_high)))
    return idx_typical, idx_low, idx_high


def select_peak_matched(paths: np.ndarray, emp_xi: np.ndarray) -> int:
    """Trajectory that best matches the empirical max and min (extrema)."""
    emp_max, emp_min = emp_xi.max(), emp_xi.min()
    path_max = paths.max(axis=1)
    path_min = paths.min(axis=1)
    score = np.abs(path_max - emp_max) + np.abs(path_min - emp_min)
    return int(np.argmin(score))


def plot_multiple(market: str):
    zemp = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    zmc = np.load(DATA / "mdn_paths" / f"{market}_mdn_paths_v2.npz", allow_pickle=True)
    xi_emp = zemp["xi"]
    paths = zmc["paths"]

    i1, i2, i3 = select_three_simulated(paths, xi_emp)
    T = min(len(xi_emp), paths.shape[1])
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    # 3 simulated trajectories first (background)
    ax.plot(t, paths[i1, :T], color="C2", lw=0.5, alpha=0.75)
    ax.plot(t, paths[i2, :T], color="C0", lw=0.5, alpha=0.75)
    ax.plot(t, paths[i3, :T], color="C3", lw=0.5, alpha=0.75)
    # Observed last, on top, thicker
    ax.plot(t, xi_emp[:T], color="black", lw=0.9)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.6)
    ax.set_xlabel("$t$ (days)", fontsize=9)
    ax.set_ylabel(r"$\xi_t$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = FIGURES / f"trajectories_v2_{market}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    (ROOT / out.name).write_bytes(out.read_bytes())
    print(f"[{market}] multiple: saved {out.name}")


def plot_peak_matched(market: str):
    zemp = np.load(DATA / "detrended" / f"{market}_detrended.npz", allow_pickle=True)
    zmc = np.load(DATA / "mdn_paths" / f"{market}_mdn_paths_v2.npz", allow_pickle=True)
    xi_emp = zemp["xi"]
    paths = zmc["paths"]

    i = select_peak_matched(paths, xi_emp)
    T = min(len(xi_emp), paths.shape[1])
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.plot(t, paths[i, :T], color="purple", lw=0.8, alpha=0.85)
    ax.plot(t, xi_emp[:T], color="black", lw=0.9)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.6)
    ax.set_xlabel("$t$ (days)", fontsize=9)
    ax.set_ylabel(r"$\xi_t$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = FIGURES / f"peak_matched_v2_{market}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    (ROOT / out.name).write_bytes(out.read_bytes())
    print(f"[{market}] peak-matched: saved {out.name}  idx={i}")


def main():
    for m in MARKETS:
        plot_multiple(m)
        plot_peak_matched(m)


if __name__ == "__main__":
    main()
