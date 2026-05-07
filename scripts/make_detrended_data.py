#!/usr/bin/env python3
"""Calcola LOESS-detrended xi_t + log-returns r_t per i 4 mercati.

Sorgenti (data/):
    gas_1826.dat   -> psv
    power_1826.dat -> pun
    pjm_2451.dat   -> pjm
    wti_2501.dat   -> wti

Output:
    data/detrended/{market}_detrended.npz    -> arrays: prices, log_prices, trend, xi, r
    data/detrended/{market}_detrended.csv    -> date-less CSV per ispezione manuale
    data/detrended/summary.txt               -> statistiche riassuntive

LOESS: stesso kernel Gaussiano di generate_mc_paths.py, frac=0.1 (default paper).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "detrended"

COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}


def load_dat(path: Path) -> np.ndarray:
    prices = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prices.append(float(line.replace(",", ".")))
    return np.array(prices, dtype=np.float64)


def loess_detrend(log_prices: np.ndarray, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel LOESS trend + residuo xi_t = log P_t - trend_t.

    Return: (trend, xi)
    """
    n = len(log_prices)
    window = max(5, int(n * frac) | 1)   # forza dispari
    half_w = window // 2
    x = np.arange(-half_w, half_w + 1)
    sigma = half_w / 2
    weights = np.exp(-x ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()
    padded = np.pad(log_prices, half_w, mode="edge")
    trend = np.array([np.sum(padded[i:i + window] * weights) for i in range(n)])
    xi = log_prices - trend
    return trend, xi


def describe(arr: np.ndarray) -> dict:
    from scipy import stats as spstats
    m, s = float(arr.mean()), float(arr.std())
    z = (arr - m) / s if s > 0 else arr * 0
    return dict(
        n=len(arr),
        mean=m, std=s,
        skew=float(spstats.skew(arr)),
        kurt=float(spstats.kurtosis(arr, fisher=False)),   # Pearson kurtosis (gaussiana=3)
        min=float(arr.min()), max=float(arr.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--frac", type=float, default=0.1)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 92)
    lines.append(f"  LOESS detrending (frac={args.frac}) -- xi_t = log P_t - trend_t,  r_t = xi_t - xi_{{t-1}}")
    lines.append("=" * 92)
    header = f"  {'market':<6}  {'N':>5} {'xi_mean':>10} {'xi_std':>10} {'xi_kurt':>10}   {'r_mean':>11} {'r_std':>10} {'r_skew':>10} {'r_kurt':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for market in args.markets.split(","):
        src = DATA / COMMODITY_FILES[market]
        if not src.exists():
            print(f"[{market}] SKIP (missing {src})")
            continue
        prices = load_dat(src)
        log_p = np.log(prices)
        trend, xi = loess_detrend(log_p, frac=args.frac)
        r = np.diff(xi)

        xi_stats = describe(xi)
        r_stats = describe(r)

        lines.append(
            f"  {market:<6}  {len(prices):>5} "
            f"{xi_stats['mean']:>+10.5f} {xi_stats['std']:>10.5f} {xi_stats['kurt']:>10.3f}   "
            f"{r_stats['mean']:>+11.6f} {r_stats['std']:>10.5f} {r_stats['skew']:>+10.4f} {r_stats['kurt']:>10.3f}"
        )

        npz_path = OUT / f"{market}_detrended.npz"
        np.savez_compressed(
            npz_path,
            prices=prices.astype(np.float64),
            log_prices=log_p.astype(np.float64),
            trend=trend.astype(np.float64),
            xi=xi.astype(np.float64),
            r=r.astype(np.float64),
            loess_frac=args.frac,
            market=market,
            source=COMMODITY_FILES[market],
        )

        csv_path = OUT / f"{market}_detrended.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("t,price,log_price,trend,xi,r\n")
            for t in range(len(prices)):
                r_t = r[t-1] if t > 0 else 0.0
                f.write(f"{t},{prices[t]:.6f},{log_p[t]:.8f},{trend[t]:.8f},{xi[t]:.8f},{r_t:.8f}\n")

        print(f"[{market}] saved {npz_path.name} + {csv_path.name}")

    lines.append("=" * 92)
    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n[summary: {(OUT/'summary.txt').relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
