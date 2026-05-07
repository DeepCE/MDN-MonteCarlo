#!/usr/bin/env python3
"""Hill tail index + QQ plot envelope + Wasserstein-1 + KS per mercato.

Risposta R1 punto 6c (tail fit quantitativo) + R1 punto 8a (GOF Fig. 7).

Per ciascun mercato:
  1. Hill estimator su |r_t| upper tail (top k order statistics) per empirico
     e per ciascuna delle 5000 MC paths
     -> distribuzione del tail index attraverso ensemble
     -> banda [P5, P95] e confronto con tail index empirico
  2. QQ plot empirico vs ensemble: per ogni quantile q da 0.01 a 0.99,
     raccogliere q-quantile da ciascuna traiettoria
     -> envelope [P5, P95] di q-quantili simulati
     -> overlay con q-quantile empirico
  3. Wasserstein-1 distance e KS statistic tra r_emp e pooled r_sim (9M samples)

Output:
  data/hill_qq_summary.txt
  data/hill_qq/<market>_hill_qq.npz  (per plot)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MDN_PATHS = DATA / "mdn_paths"
OUT = DATA / "hill_qq"
OUT.mkdir(parents=True, exist_ok=True)


def hill_estimator(x: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index xi (= 1/alpha) on top-k order statistics of |x|."""
    x_abs = np.abs(x)
    x_sorted = np.sort(x_abs)[::-1]  # descending
    if k >= len(x_sorted):
        k = len(x_sorted) - 1
    if k <= 0:
        return float("nan")
    threshold = x_sorted[k]
    top = x_sorted[:k]
    if threshold <= 0 or (top <= 0).any():
        return float("nan")
    return float(np.mean(np.log(top) - np.log(threshold)))


def choose_k(n: int) -> int:
    """Heuristic: k ~ sqrt(n) rounded (standard for Hill on ~thousands of obs)."""
    return int(max(20, round(np.sqrt(n))))


def qq_bands(paths: np.ndarray, q_levels: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per ciascuna traiettoria di paths (N, T): calcolare diff -> r,
    quindi q-quantili di r per ogni path -> matrice (N, Q).
    Ritorna (P5, P50, P95) per livello, shape (Q,).
    """
    N = paths.shape[0]
    r = np.diff(paths, axis=1)
    # Matrix of per-path quantiles
    qmat = np.percentile(r, q_levels * 100, axis=1).T  # (N, Q)
    return np.percentile(qmat, 5, axis=0), np.percentile(qmat, 50, axis=0), np.percentile(qmat, 95, axis=0)


def run_market(market: str) -> dict:
    npz_path = MDN_PATHS / f"{market}_mdn_paths.npz"
    if not npz_path.exists():
        return None
    z = np.load(npz_path, allow_pickle=True)
    paths = z["paths"]
    xi_emp = z["empirical_xi"]
    r_emp = np.diff(xi_emp)
    N, T = paths.shape

    # --- Hill tail index
    k = choose_k(len(r_emp))
    hill_emp = hill_estimator(r_emp, k)

    # Per-path hill
    hills_sim = np.array([hill_estimator(np.diff(paths[i]), k) for i in range(N)])
    hill_band = (float(np.percentile(hills_sim, 5)),
                 float(np.percentile(hills_sim, 50)),
                 float(np.percentile(hills_sim, 95)))
    hill_in = hill_band[0] <= hill_emp <= hill_band[2]

    # --- QQ envelope
    q_levels = np.concatenate([
        np.array([0.005, 0.01, 0.025]),
        np.arange(0.05, 1.0, 0.05),
        np.array([0.975, 0.99, 0.995]),
    ])
    q_emp = np.percentile(r_emp, q_levels * 100)
    qq_lo, qq_med, qq_hi = qq_bands(paths, q_levels)
    qq_in = (qq_lo <= q_emp) & (q_emp <= qq_hi)
    qq_coverage = float(qq_in.mean())

    # --- Wasserstein + KS (pooled)
    # Subsample to keep compute manageable
    rng = np.random.default_rng(42)
    n_sub = min(200_000, N * (T - 1))
    r_sim_pooled = np.diff(paths, axis=1).ravel()
    if len(r_sim_pooled) > n_sub:
        idx = rng.choice(len(r_sim_pooled), size=n_sub, replace=False)
        r_sim_pooled = r_sim_pooled[idx]
    w1 = float(wasserstein_distance(r_emp, r_sim_pooled))
    ks_stat, ks_p = ks_2samp(r_emp, r_sim_pooled)

    # Save arrays for plots
    out = OUT / f"{market}_hill_qq.npz"
    np.savez_compressed(
        out,
        hill_emp=hill_emp, hills_sim=hills_sim, hill_band=np.array(hill_band), hill_in=hill_in,
        q_levels=q_levels, q_emp=q_emp, qq_lo=qq_lo, qq_med=qq_med, qq_hi=qq_hi,
        qq_in=qq_in, qq_coverage=qq_coverage,
        w1=w1, ks_stat=ks_stat, ks_p=ks_p, k=k,
    )

    return dict(
        k=k,
        hill_emp=hill_emp, hill_band=hill_band, hill_in=hill_in,
        qq_coverage=qq_coverage,
        w1=w1, ks_stat=float(ks_stat), ks_p=float(ks_p),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    args = ap.parse_args()

    lines = []
    lines.append("=" * 106)
    lines.append("  HILL TAIL INDEX + QQ ENVELOPE + WASSERSTEIN + KS")
    lines.append("=" * 106)
    lines.append(
        f"  {'market':<6} {'k':>5} {'hill_emp':>10} {'hill_P5':>10} {'hill_P50':>10} {'hill_P95':>10} "
        f"{'in_band':>8} {'qq_cov':>8} {'W1':>9} {'KS':>8} {'p_KS':>8}"
    )
    for market in args.markets.split(","):
        r = run_market(market)
        if r is None:
            print(f"[{market}] SKIP")
            continue
        hill_band = r["hill_band"]
        lines.append(
            f"  {market:<6} {r['k']:>5d} {r['hill_emp']:>10.4f} "
            f"{hill_band[0]:>10.4f} {hill_band[1]:>10.4f} {hill_band[2]:>10.4f} "
            f"{str(bool(r['hill_in'])):>8} {r['qq_coverage']:>8.2f} "
            f"{r['w1']:>9.5f} {r['ks_stat']:>8.4f} {r['ks_p']:>8.4f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Hill tail index xi: larger => heavier tail. Compare empirical vs simulated ensemble band.")
    lines.append("  QQ coverage: frazione di q-quantili empirici che cadono in banda [P5, P95] simulata.")
    lines.append("  W1: Wasserstein-1 distance r_emp vs pooled r_sim (lower => closer).")
    lines.append("  KS: Kolmogorov-Smirnov stat (not rejected if p > 0.05 => distributions indistinguishable).")
    lines.append("=" * 106)
    text = "\n".join(lines)
    print(text)
    out = DATA / "hill_qq_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[summary: {out.name}]")


if __name__ == "__main__":
    main()
