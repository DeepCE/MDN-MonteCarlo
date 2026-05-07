#!/usr/bin/env python3
"""Bande [P5, P95] per momenti di r_t, per alpha mean reversion, per ACF^2.

Per ciascun mercato:
  1. Carica 5000 MC paths MDN da data/mdn_paths/<market>_mdn_paths.npz
  2. Per ciascuna traiettoria MC: calcola std/skew/kurt di r_t (lo stesso che empiricamente)
  3. Costruisce bande [P5, P50, P95] da distribuzione cross-trajectory
  4. Confronta con empirico e marca ✓/✗ inclusione in banda

Risposta diretta a R1 punto 6 (bootstrap comparison), R1 punto 11 (CI bands),
R1 punto 7 (alpha SE + test simulato).

Output:
  data/bands_summary.txt   tabella markdown-like
  data/bands_data.npz      array per figure successive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MDN_PATHS = DATA / "mdn_paths"

COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}

# Suffisso applicato al nome npz letto da mdn_paths/ per la variante scelta.
VARIANT_SUFFIX = {
    "retrained": "",
    "paper_k8": "_paper_k8",
    "paper_tab3": "_paper_tab3",
    "v2": "_v2",
}


def load_dat(path: Path) -> np.ndarray:
    prices = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prices.append(float(line.replace(",", ".")))
    return np.array(prices)


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


def per_path_moments(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per ogni riga di paths (shape: N x T) calcola std/skew/kurt di diff(path).

    Returns: (mean, std, skew, kurt) ciascuno shape (N,).
    """
    r = np.diff(paths, axis=1)
    m = r.mean(axis=1)
    s = r.std(axis=1)
    s_safe = np.where(s > 0, s, 1.0)
    z = (r - m[:, None]) / s_safe[:, None]
    skew = (z ** 3).mean(axis=1)
    kurt = (z ** 4).mean(axis=1)
    return m, s, skew, kurt


def alpha_from_series(xi: np.ndarray) -> tuple[float, float]:
    """Fit OLS r_t = c - alpha * xi_{t-1} + eps ; return (alpha, SE(alpha))."""
    x = xi[:-1]
    r = np.diff(xi)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    c_hat, slope = beta  # slope = -alpha
    alpha = -slope
    resid = r - X @ beta
    dof = len(r) - 2
    sigma2 = float(np.sum(resid ** 2) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    var_slope = sigma2 * XtX_inv[1, 1]
    return float(alpha), float(np.sqrt(var_slope))


def per_path_alpha(paths: np.ndarray) -> np.ndarray:
    """Alpha per ogni traiettoria (vettorizzato)."""
    N, T = paths.shape
    x = paths[:, :-1]
    r = np.diff(paths, axis=1)
    # Regression per row: slope = (cov(x, r)) / var(x) adjusting for mean
    x_mean = x.mean(axis=1, keepdims=True)
    r_mean = r.mean(axis=1, keepdims=True)
    xc = x - x_mean
    rc = r - r_mean
    num = (xc * rc).sum(axis=1)
    den = (xc * xc).sum(axis=1)
    slope = num / np.where(den > 0, den, 1.0)
    return -slope


def acf_sq_empirical(r: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """ACF of r^2 for lags 1..max_lag."""
    r2 = r ** 2
    r2_c = r2 - r2.mean()
    denom = (r2_c ** 2).sum()
    return np.array([
        (r2_c[:-k] * r2_c[k:]).sum() / denom if k > 0 else 1.0
        for k in range(1, max_lag + 1)
    ])


def per_path_acf_sq(paths: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """ACF^2 per ogni path, shape (N, max_lag)."""
    N = paths.shape[0]
    out = np.empty((N, max_lag), dtype=np.float64)
    for i in range(N):
        r = np.diff(paths[i])
        out[i] = acf_sq_empirical(r, max_lag=max_lag)
    return out


def acf_level(x: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """ACF of a level series (xi_t), lag 1..max_lag. Used for half-life."""
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return np.array([
        (xc[:-k] * xc[k:]).sum() / denom
        for k in range(1, max_lag + 1)
    ])


def half_life_from_acf(acf: np.ndarray) -> float:
    """Half-life from linear interpolation of the first lag where ACF <= 0.5.

    Returns np.inf if ACF never crosses 0.5 within max_lag.
    Distribution-free: does NOT assume any OU / AR(1) model.
    """
    lags = np.arange(1, len(acf) + 1)
    # find first k where acf[k-1] <= 0.5
    below = np.where(acf <= 0.5)[0]
    if len(below) == 0:
        return float("inf")
    k_star = below[0]
    if k_star == 0:
        return 1.0   # already below at lag 1
    # linear interpolation between lag k_star and k_star+1 (0-indexed -> k_star, k_star+1)
    y1, y2 = acf[k_star - 1], acf[k_star]
    x1, x2 = lags[k_star - 1], lags[k_star]
    if y1 == y2:
        return float(x1)
    return float(x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1))


def per_path_acf_level(paths: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """ACF of xi_t (levels) for each path, shape (N, max_lag)."""
    N = paths.shape[0]
    out = np.empty((N, max_lag), dtype=np.float64)
    for i in range(N):
        out[i] = acf_level(paths[i], max_lag=max_lag)
    return out


def per_path_half_life(paths: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """Half-life per path from ACF(xi) interpolation."""
    N = paths.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        a = acf_level(paths[i], max_lag=max_lag)
        out[i] = half_life_from_acf(a)
    return out


def band(vals: np.ndarray, probs=(5, 50, 95)) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]   # escape inf/nan from ill-defined HL
    return tuple(float(np.percentile(vals, p)) for p in probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--max_acf_lag", type=int, default=30)
    ap.add_argument("--max_acf_xi_lag", type=int, default=60,
                    help="max lag per ACF(xi) usata per half-life")
    ap.add_argument("--variant", choices=sorted(VARIANT_SUFFIX.keys()), default="retrained",
                    help="quale set di paths analizzare")
    ap.add_argument("--out-suffix", default=None,
                    help="override del suffisso output txt/npz; default = VARIANT_SUFFIX[variant]")
    ap.add_argument("--percentiles", default="5,50,95",
                    help="percentili delle bande in formato 'P_low,P_median,P_high' (default: 5,50,95)")
    args = ap.parse_args()

    in_suffix = VARIANT_SUFFIX[args.variant]
    out_suffix = args.out_suffix if args.out_suffix is not None else in_suffix

    p_lo, p_mid, p_hi = (float(x) for x in args.percentiles.split(","))

    def band_fn(vals):
        vals = vals[np.isfinite(vals)]
        return tuple(float(np.percentile(vals, p)) for p in (p_lo, p_mid, p_hi))

    lines = []
    lines.append("=" * 108)
    lines.append(f"  BANDS [P{p_lo:g}, P{p_mid:g}, P{p_hi:g}] FROM MC TRAJECTORIES  (variant: {args.variant})")
    lines.append("  Mean reversion metric: half-life from ACF(xi) crossing 0.5  [distribution-free,")
    lines.append("  does not assume OU; alpha OLS reported for reference only, see Mari & Mari 2026]")
    lines.append("=" * 108)

    bands_dict = {}

    for market in args.markets.split(","):
        npz_path = MDN_PATHS / f"{market}_mdn_paths{in_suffix}.npz"
        if not npz_path.exists():
            print(f"[{market}] SKIP (no paths at {npz_path})")
            continue
        print(f"\n[{market}] loading {npz_path.name} ...")
        z = np.load(npz_path, allow_pickle=True)
        paths = z["paths"]
        xi_emp = z["empirical_xi"]
        r_emp = np.diff(xi_emp)

        # Empirical moments
        m_e = float(r_emp.mean())
        s_e = float(r_emp.std())
        zz = (r_emp - m_e) / s_e
        skew_e = float((zz ** 3).mean())
        kurt_e = float((zz ** 4).mean())

        # Mean reversion: half-life from ACF(xi) + alpha OLS (reference)
        acf_xi_emp = acf_level(xi_emp, max_lag=args.max_acf_xi_lag)
        hl_emp = half_life_from_acf(acf_xi_emp)
        alpha_e, alpha_e_se = alpha_from_series(xi_emp)

        # Per-path moments + mean reversion
        m_p, s_p, skew_p, kurt_p = per_path_moments(paths)
        alpha_p = per_path_alpha(paths)
        acf_xi_paths = per_path_acf_level(paths, max_lag=args.max_acf_xi_lag)
        hl_paths = per_path_half_life(paths, max_lag=args.max_acf_xi_lag)

        # Bands
        b_mean = band_fn(m_p)
        b_std = band_fn(s_p)
        b_skew = band_fn(skew_p)
        b_kurt = band_fn(kurt_p)
        b_alpha = band_fn(alpha_p)
        b_hl = band_fn(hl_paths)

        def check(emp, b):
            return "OK" if b[0] <= emp <= b[2] else "OUT"

        n_path = paths.shape[0]
        lines.append("")
        lines.append(f"-- {market.upper()}  (N_path={n_path}, T={paths.shape[1]}) "
                     + "-" * max(1, 40 - len(market)))
        plo_lbl = f"P{p_lo:g}"; pmid_lbl = f"P{p_mid:g}"; phi_lbl = f"P{p_hi:g}"
        lines.append(f"  {'quantity':<12} {'emp':>12} {plo_lbl:>12} {pmid_lbl:>12} {phi_lbl:>12}   in_band")
        lines.append(f"  {'mean':<12} {m_e:>+12.5f} {b_mean[0]:>+12.5f} {b_mean[1]:>+12.5f} {b_mean[2]:>+12.5f}   {check(m_e, b_mean)}")
        lines.append(f"  {'std':<12} {s_e:>12.5f} {b_std[0]:>12.5f} {b_std[1]:>12.5f} {b_std[2]:>12.5f}   {check(s_e, b_std)}")
        lines.append(f"  {'skew':<12} {skew_e:>+12.4f} {b_skew[0]:>+12.4f} {b_skew[1]:>+12.4f} {b_skew[2]:>+12.4f}   {check(skew_e, b_skew)}")
        lines.append(f"  {'kurt':<12} {kurt_e:>12.3f} {b_kurt[0]:>12.3f} {b_kurt[1]:>12.3f} {b_kurt[2]:>12.3f}   {check(kurt_e, b_kurt)}")
        lines.append(f"  {'half-life':<12} {hl_emp:>12.2f} {b_hl[0]:>12.2f} {b_hl[1]:>12.2f} {b_hl[2]:>12.2f}   {check(hl_emp, b_hl)}")
        lines.append(f"    [ref only] alpha_OLS emp = {alpha_e:+.5f} (SE {alpha_e_se:.5f}); "
                     f"sim [{b_alpha[0]:+.4f}, {b_alpha[1]:+.4f}, {b_alpha[2]:+.4f}]")

        # ACF^2 (r_t)
        acf_emp = acf_sq_empirical(r_emp, max_lag=args.max_acf_lag)
        acf_paths = per_path_acf_sq(paths, max_lag=args.max_acf_lag)
        acf_bands = np.percentile(acf_paths, [p_lo, p_mid, p_hi], axis=0)

        in_band = (acf_bands[0] <= acf_emp) & (acf_emp <= acf_bands[2])
        coverage = float(in_band.mean())
        lines.append(f"  ACF^2(r) lag 1..{args.max_acf_lag}: empirical in band for "
                     f"{int(in_band.sum())}/{args.max_acf_lag} lags ({coverage*100:.0f}%)")

        # ACF(xi) — coverage empirical inside per-path band
        acf_xi_bands = np.percentile(acf_xi_paths, [p_lo, p_mid, p_hi], axis=0)
        in_band_xi = (acf_xi_bands[0] <= acf_xi_emp) & (acf_xi_emp <= acf_xi_bands[2])
        coverage_xi = float(in_band_xi.mean())
        lines.append(f"  ACF(xi) lag 1..{args.max_acf_xi_lag}: empirical in band for "
                     f"{int(in_band_xi.sum())}/{args.max_acf_xi_lag} lags ({coverage_xi*100:.0f}%)")

        bands_dict[market] = dict(
            r_emp_mean=m_e, r_emp_std=s_e, r_emp_skew=skew_e, r_emp_kurt=kurt_e,
            alpha_emp=alpha_e, alpha_emp_se=alpha_e_se,
            hl_emp=hl_emp,
            b_mean=b_mean, b_std=b_std, b_skew=b_skew, b_kurt=b_kurt,
            b_alpha=b_alpha, b_hl=b_hl,
            acf_emp=acf_emp, acf_bands=acf_bands, acf_coverage=coverage,
            acf_xi_emp=acf_xi_emp, acf_xi_bands=acf_xi_bands, acf_xi_coverage=coverage_xi,
            percentiles=(p_lo, p_mid, p_hi),
        )

    lines.append("")
    lines.append("=" * 100)
    text = "\n".join(lines)
    print("\n" + text)

    out_txt = DATA / f"bands_summary{out_suffix}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    out_npz = DATA / f"bands_data{out_suffix}.npz"
    np.savez_compressed(out_npz, **{f"{k}_{kk}": vv for k, d in bands_dict.items()
                                    for kk, vv in d.items()})
    print(f"\n[summary: {out_txt.name}]  [data: {out_npz.name}]")


if __name__ == "__main__":
    main()
