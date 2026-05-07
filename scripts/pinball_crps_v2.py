#!/usr/bin/env python3
"""Pinball loss e CRPS marginali per MDN v2 vs DeepAR, 4 mercati.

Risposta R2.3 (quantile forecast comparison).

Metriche:
  - CRPS marginale: CRPS(F_sim, y_emp) dove F_sim e' la distribuzione marginale
    dei returns simulati pooled (N_traj x (T-1)) e y_emp i returns empirici.
    Approssimazione via energy form: CRPS(F, y) = E|X-y| - 0.5 * E|X-X'|
    dove X, X' ~ F i.i.d.
  - Pinball loss al quantile tau: L_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q))
    dove q e' il tau-quantile della distribuzione simulata.

Le metriche sono pooled marginali (non condizionali al tempo), coerenti con il
framework di validation per-path del paper e con il fatto che entrambi i modelli
(MDN e DeepAR) producono paths autoregressivi indipendenti, non forecast
condizionati sulla storia empirica al tempo t.

Input:
  data/mdn_paths/<market>_mdn_paths_v2.npz         MDN paths
  data/benchmarks/<market>_deepar_paths.npz        DeepAR paths
  data/detrended/<market>_detrended.npz            empirical xi + r

Output:
  data/pinball_crps_summary.txt                    Tabella MDN vs DeepAR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MARKETS = ("psv", "pun", "pjm", "wti")
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def load_data(market: str) -> dict:
    """Load empirical + MDN + DeepAR aligned on the DeepAR test window.

    DeepAR paths span only the last T_test steps (out-of-sample). We align
    all three quantities to this test window:
      - empirical r_emp: diff of test_xi   shape (T_test-1,)
      - DeepAR r_sim   : diff of paths    shape (500, T_test-1)
      - MDN r_sim      : diff of last T_test steps of MDN paths
                                          shape (5000, T_test-1)
    """
    # DeepAR: defines test window
    z_d = np.load(DATA / "benchmarks" / f"{market}_deepar_paths.npz", allow_pickle=True)
    deepar_paths_xi = z_d["paths"]           # (500, T_test) detrended xi
    test_xi = z_d["test_xi"]                  # (T_test,)
    T_test = len(test_xi)

    # Empirical on test window
    emp_r = np.diff(test_xi)                  # (T_test - 1,)

    # DeepAR returns on same window
    deepar_r = np.diff(deepar_paths_xi, axis=1)   # (500, T_test - 1)

    # MDN: last T_test steps of simulated xi
    z_m = np.load(DATA / "mdn_paths" / f"{market}_mdn_paths_v2.npz", allow_pickle=True)
    mdn_paths_xi = z_m["paths"]               # (5000, T)
    mdn_tail = mdn_paths_xi[:, -T_test:]      # (5000, T_test)
    mdn_r = np.diff(mdn_tail, axis=1)         # (5000, T_test - 1)

    return {
        "market": market,
        "T_test": T_test,
        "emp_r": emp_r,
        "mdn_r": mdn_r,
        "deepar_r": deepar_r,
    }


def crps_energy(sim_pooled: np.ndarray, y_emp: np.ndarray,
                subsample: int = 50_000, rng: np.random.Generator | None = None
                ) -> float:
    """CRPS via energy formula: E|X-y| - 0.5 * E|X-X'|.

    Subsample sim_pooled for tractable O(N^2) second term.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(sim_pooled) > subsample:
        idx = rng.choice(len(sim_pooled), size=subsample, replace=False)
        X = sim_pooled[idx]
    else:
        X = sim_pooled

    # First term: mean over y of E_X |X - y|
    # We sort X once and use it for all y (vectorized across y would be N*|y|)
    # For large X and y, this is the bottleneck. Use chunks.
    chunk = 500
    first_terms = np.empty(len(y_emp), dtype=np.float64)
    for start in range(0, len(y_emp), chunk):
        yc = y_emp[start:start + chunk]
        # |X - y|: (len(X), len(yc))
        first_terms[start:start + chunk] = np.mean(np.abs(X[:, None] - yc[None, :]),
                                                    axis=0)

    # Second term: 0.5 * E|X - X'|
    # Use subsample of pairs: draw N2 pairs
    N2 = min(100_000, len(X) * len(X))
    i = rng.integers(0, len(X), size=N2)
    j = rng.integers(0, len(X), size=N2)
    second = 0.5 * np.mean(np.abs(X[i] - X[j]))

    crps = np.mean(first_terms) - second
    return float(crps)


def pinball(sim_pooled: np.ndarray, y_emp: np.ndarray, tau: float) -> float:
    q = np.quantile(sim_pooled, tau)
    diff = y_emp - q
    loss = np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    return float(np.mean(loss))


def metrics_for_market(sim: np.ndarray, emp: np.ndarray) -> dict:
    """Compute CRPS and pinball@5 quantiles. sim: (N, T-1) or (N*(T-1),). emp: (T-1,)."""
    if sim.ndim == 2:
        sim_pool = sim.ravel()
    else:
        sim_pool = sim
    rng = np.random.default_rng(42)
    out = {"CRPS": crps_energy(sim_pool, emp, rng=rng)}
    for tau in QUANTILES:
        out[f"pb@{tau:.2f}"] = pinball(sim_pool, emp, tau)
    out["pb_avg"] = float(np.mean([out[f"pb@{tau:.2f}"] for tau in QUANTILES]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    args = ap.parse_args()

    rows_mdn = {}
    rows_deepar = {}
    T_tests = {}
    for m in args.markets.split(","):
        print(f"\n[{m}] ...", flush=True)
        d = load_data(m)
        emp = d["emp_r"]
        T_tests[m] = d["T_test"]
        print(f"  T_test={d['T_test']}  emp: {emp.shape}  mdn: {d['mdn_r'].shape}  "
              f"deepar: {d['deepar_r'].shape}")
        rows_mdn[m] = metrics_for_market(d["mdn_r"], emp)
        print(f"  MDN-v2 : CRPS={rows_mdn[m]['CRPS']:.5f}  pb_avg={rows_mdn[m]['pb_avg']:.5f}")
        rows_deepar[m] = metrics_for_market(d["deepar_r"], emp)
        print(f"  DeepAR : CRPS={rows_deepar[m]['CRPS']:.5f}  pb_avg={rows_deepar[m]['pb_avg']:.5f}")

    # Summary table
    lines = []
    lines.append("=" * 120)
    lines.append("  PINBALL + CRPS -- pooled marginal, MDN v2 vs DeepAR")
    lines.append("=" * 120)
    header = (f"  {'market':<6} {'model':<8}   "
              f"{'CRPS':>9}   {'pb@0.05':>9} {'pb@0.25':>9} {'pb@0.50':>9} {'pb@0.75':>9} {'pb@0.95':>9}   {'pb_avg':>9}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for m in args.markets.split(","):
        for label, d in (("MDN-v2", rows_mdn.get(m)), ("DeepAR", rows_deepar.get(m))):
            if d is None:
                lines.append(f"  {m:<6} {label:<8}   {'n/a':>9}")
                continue
            lines.append(
                f"  {m:<6} {label:<8}   "
                f"{d['CRPS']:>9.5f}   "
                + " ".join(f"{d[f'pb@{t:.2f}']:>9.5f}" for t in QUANTILES)
                + f"   {d['pb_avg']:>9.5f}"
            )
        lines.append("")

    lines.append("=" * 120)
    lines.append("  Lower is better for all metrics. Pooled marginal comparison: the simulated")
    lines.append("  ensemble is treated as a sample from the marginal return distribution,")
    lines.append("  independently of time (consistent with the per-path moment validation).")
    lines.append("=" * 120)

    text = "\n".join(lines)
    print("\n" + text)
    out = DATA / "pinball_crps_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[summary: {out.name}]")


if __name__ == "__main__":
    main()
