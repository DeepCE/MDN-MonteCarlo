#!/usr/bin/env python3
"""Goodness-of-fit quantitative metrics for the unconditional distribution (R1.8.a).

For each of the four markets:
  - Empirical r = detrended log-returns
  - Simulated r (MDN v2, pooled from 5000 MC paths)
  - Gaussian fit to empirical r
  - Student-t fit to empirical r

Computes Wasserstein-1 distance and Kolmogorov-Smirnov statistic between
the empirical distribution and each of the three candidate distributions.
MDN: two-sample W1 and KS against empirical (sample-vs-sample).
Gaussian/Student-t: sample-vs-parametric W1 approximated by large simulation,
KS via one-sample.

Output: data/wasserstein_ks_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MARKETS = ("psv", "pun", "pjm", "wti")

N_PARAM_SAMPLES = 200_000   # parametric resample for W1 vs empirical
RNG = np.random.default_rng(42)


def load_empirical(m: str) -> np.ndarray:
    z = np.load(DATA / "detrended" / f"{m}_detrended.npz", allow_pickle=True)
    return z["r"]


def load_mdn_pool(m: str) -> np.ndarray:
    z = np.load(DATA / "mdn_paths" / f"{m}_mdn_paths_v2.npz", allow_pickle=True)
    return np.diff(z["paths"], axis=1).ravel()


def metrics_mdn(emp: np.ndarray, sim: np.ndarray, subsample: int = 100_000) -> dict:
    """Two-sample W1 and KS between empirical and pooled MDN simulated."""
    if len(sim) > subsample:
        idx = RNG.choice(len(sim), size=subsample, replace=False)
        sim_s = sim[idx]
    else:
        sim_s = sim
    w1 = stats.wasserstein_distance(emp, sim_s)
    ks, p_ks = stats.ks_2samp(emp, sim_s)
    return {"W1": float(w1), "KS": float(ks), "KS_p": float(p_ks)}


def metrics_gaussian(emp: np.ndarray) -> dict:
    """W1 sample-vs-parametric-sample; KS one-sample against N(mu_hat, sigma_hat)."""
    mu, sigma = float(emp.mean()), float(emp.std(ddof=1))
    ref = RNG.normal(mu, sigma, size=N_PARAM_SAMPLES)
    w1 = stats.wasserstein_distance(emp, ref)
    ks, p_ks = stats.kstest(emp, "norm", args=(mu, sigma))
    return {"W1": float(w1), "KS": float(ks), "KS_p": float(p_ks)}


def metrics_tstudent(emp: np.ndarray) -> dict:
    """W1 sample-vs-t-resample; KS one-sample against t(df, loc, scale)."""
    df, loc, scale = stats.t.fit(emp)
    ref = stats.t.rvs(df, loc=loc, scale=scale, size=N_PARAM_SAMPLES, random_state=RNG)
    w1 = stats.wasserstein_distance(emp, ref)
    ks, p_ks = stats.kstest(emp, "t", args=(df, loc, scale))
    return {"W1": float(w1), "KS": float(ks), "KS_p": float(p_ks), "df": float(df)}


def main():
    rows = []
    for m in MARKETS:
        emp = load_empirical(m)
        sim = load_mdn_pool(m)
        n_emp, n_sim = len(emp), len(sim)
        print(f"[{m}] N_emp={n_emp}  N_sim={n_sim}  (pool before subsample)")

        mdn = metrics_mdn(emp, sim)
        gau = metrics_gaussian(emp)
        tst = metrics_tstudent(emp)
        rows.append((m, n_emp, n_sim, mdn, gau, tst))
        print(f"  MDN    : W1={mdn['W1']:.6f}  KS={mdn['KS']:.4f}  KS_p={mdn['KS_p']:.4f}")
        print(f"  Gauss  : W1={gau['W1']:.6f}  KS={gau['KS']:.4f}  KS_p={gau['KS_p']:.4f}")
        print(f"  t-Stud : W1={tst['W1']:.6f}  KS={tst['KS']:.4f}  KS_p={tst['KS_p']:.4f}  "
              f"df={tst['df']:.2f}")

    lines = []
    lines.append("=" * 116)
    lines.append("  WASSERSTEIN-1 and KOLMOGOROV-SMIRNOV for unconditional return distribution")
    lines.append("  (response to R1.8.a: quantitative GOF for Figure 7)")
    lines.append("=" * 116)
    lines.append(f"  {'market':<6} {'N_emp':>5} "
                 f"{'W1_MDN':>10} {'W1_Gauss':>10} {'W1_tStud':>10}   "
                 f"{'KS_MDN':>8} {'KS_Gauss':>9} {'KS_tStud':>9}   "
                 f"{'p_MDN':>7} {'p_Gauss':>8} {'p_tStud':>8}   "
                 f"{'df_t':>6}")
    for m, n_emp, n_sim, mdn, gau, tst in rows:
        lines.append(
            f"  {m:<6} {n_emp:>5d} "
            f"{mdn['W1']:>10.6f} {gau['W1']:>10.6f} {tst['W1']:>10.6f}   "
            f"{mdn['KS']:>8.4f} {gau['KS']:>9.4f} {tst['KS']:>9.4f}   "
            f"{mdn['KS_p']:>7.4f} {gau['KS_p']:>8.4f} {tst['KS_p']:>8.4f}   "
            f"{tst['df']:>6.2f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  W1  (Wasserstein-1, lower is better): earth mover's distance between")
    lines.append("      empirical and candidate distribution.")
    lines.append("  KS  (Kolmogorov-Smirnov, lower is better): max |F_emp - F_ref|.")
    lines.append("  KS_p (p-value): high p-value = cannot reject the null that sample is")
    lines.append("      drawn from the reference distribution; low p-value = rejection.")
    lines.append("  df_t: fitted degrees of freedom of the Student-t reference.")
    lines.append("=" * 116)

    text = "\n".join(lines)
    print("\n" + text)
    out = DATA / "wasserstein_ks_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[summary: {out.name}]")


if __name__ == "__main__":
    main()
