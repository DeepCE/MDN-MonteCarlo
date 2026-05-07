#!/usr/bin/env python3
"""VaR backtesting MDN: Kupiec (unconditional coverage) + Christoffersen (independence).

Procedura (rolling 1-day ahead):
  Per ogni t = m..T-1:
    - Prendi context x_t = xi[t-m+1..t]
    - Usa MDN per ottenere GMM condizionale p(xi_{t+1} | x_t) -> distribuzione del return r_{t+1} = xi_{t+1} - xi_t
    - Calcola VaR_alpha al livello alpha (1%, 5%) dalla GMM trasla (mu - xi_t)
    - Osserva r_{t+1}^{emp} = xi[t+1] - xi[t]
    - Segna exceedance se r_{t+1}^{emp} < VaR_alpha  (VaR-left-tail; simmetrico per right)

Test:
  - Kupiec LR_uc: se le eccedenze in totale si conformano al livello alpha atteso
  - Christoffersen LR_ind: indipendenza delle eccedenze (no clustering)
  - LR_cc = LR_uc + LR_ind (conditional coverage joint)

Risposta R1 punto 6.d (VaR backtest).

Output:
  data/var_backtest_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdn_models import load_checkpoint_model  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}


def load_dat(path: Path) -> np.ndarray:
    with open(path) as f:
        return np.array([float(line.strip().replace(",", "."))
                         for line in f if line.strip()])


def loess_detrend(prices, frac=0.1):
    lp = np.log(prices)
    n = len(lp)
    w = max(5, int(n * frac) | 1)
    hw = w // 2
    wts = np.exp(-np.arange(-hw, hw + 1) ** 2 / (2 * (hw / 2) ** 2))
    wts /= wts.sum()
    pad = np.pad(lp, hw, mode="edge")
    trend = np.array([np.sum(pad[i:i + w] * wts) for i in range(n)])
    return lp - trend


# Model classes imported via mdn_models.load_checkpoint_model


def gmm_quantile(pi, mu, sigma, q, grid=2048):
    lo = float((mu - 6 * sigma).min())
    hi = float((mu + 6 * sigma).max())
    xs = np.linspace(lo, hi, grid)
    cdf = np.zeros_like(xs)
    for pk, mk, sk in zip(pi, mu, sigma):
        cdf += pk * norm.cdf(xs, loc=mk, scale=sk)
    idx = int(np.clip(np.searchsorted(cdf, q), 1, grid - 1))
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    return float(x0) if y1 == y0 else float(x0 + (q - y0) / (y1 - y0) * (x1 - x0))


def kupiec_lr(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec LR_uc test for unconditional coverage."""
    n = len(hits)
    x = int(hits.sum())
    phat = x / n if n > 0 else 0.0
    if phat <= 0 or phat >= 1:
        return 0.0, 1.0
    ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll_alt = x * np.log(phat) + (n - x) * np.log(1 - phat)
    lr = -2 * (ll_null - ll_alt)
    p = float(1 - chi2.cdf(lr, df=1))
    return float(lr), p


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Independence test (no clustering) via Markov chain transitions 0->1, 1->1."""
    hits = hits.astype(int)
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n0 = n00 + n01
    n1 = n10 + n11
    if n0 == 0 or n1 == 0:
        return 0.0, 1.0
    p01 = n01 / n0
    p11 = n11 / n1
    p_any = (n01 + n11) / (n0 + n1)
    if p_any <= 0 or p_any >= 1 or p01 <= 0 or p01 >= 1 or p11 <= 0 or p11 >= 1:
        return 0.0, 1.0
    ll_null = (n00 + n10) * np.log(1 - p_any) + (n01 + n11) * np.log(p_any)
    ll_alt = n00 * np.log(1 - p01) + n01 * np.log(p01) \
        + n10 * np.log(1 - p11) + n11 * np.log(p11)
    lr = -2 * (ll_null - ll_alt)
    p = float(1 - chi2.cdf(lr, df=1))
    return float(lr), p


CKPT_PATTERNS = {
    "retrained": "mdn_final_{market}.pt",
    "v2": "mdn_v2_{market}.pt",
    "pinball": "mdn_v2_{market}_pinball.pt",
    "pinball_left": "mdn_v2_{market}_pinball_left.pt",
}


def run_market(market, device, alphas=(0.01, 0.05), variant="v2", exclude_window=None):
    """exclude_window: (start_idx, end_idx) tuple of empirical-index range to exclude from backtest.
    Useful to assess the effect of a dominant outlier event (e.g. WTI Cushing 2020-04).
    """
    pattern = CKPT_PATTERNS[variant]
    ckpt_path = MODELS / pattern.format(market=market)
    if not ckpt_path.exists():
        print(f"[{market}] SKIP (no checkpoint at {ckpt_path})")
        return None
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    m = cfg["lookback"]
    K = cfg["n_components"]

    prices = load_dat(DATA / COMMODITY_FILES[market])
    xi = loess_detrend(prices)

    T_eff = len(xi) - m
    X = np.stack([xi[t:t + m] for t in range(T_eff)], axis=0).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)

    print(f"[{market}] forward: T_eff={T_eff}, m={m}, K={K}")
    with torch.no_grad():
        pis, mus, sigmas = [], [], []
        batch = 512
        for s in range(0, T_eff, batch):
            p, mu, sg = model(X_t[s:s + batch])
            pis.append(p.cpu().numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(sg.cpu().numpy())
    PI = np.concatenate(pis, axis=0)
    MU = np.concatenate(mus, axis=0)
    SIGMA = np.concatenate(sigmas, axis=0)

    # r_{t+1}^emp = xi[t+1] - xi[t]  where t ranges from m-1..N-2 (aligned with GMM at t)
    # GMM index s corresponds to context xi[s..s+m-1] predicting xi[s+m]
    # so the empirical return to compare is xi[s+m] - xi[s+m-1]
    r_emp = xi[m:] - xi[m - 1:-1]
    assert len(r_emp) == T_eff

    results = {}
    for alpha in alphas:
        # VaR on r (left tail => quantile alpha on (xi_next - xi_current))
        # Compute quantile alpha of xi_next distribution minus xi_current
        var_left = np.empty(T_eff)
        var_right = np.empty(T_eff)
        for t in range(T_eff):
            q_lo = gmm_quantile(PI[t], MU[t], SIGMA[t], alpha)
            q_hi = gmm_quantile(PI[t], MU[t], SIGMA[t], 1 - alpha)
            xi_curr = float(xi[m - 1 + t])
            var_left[t] = q_lo - xi_curr
            var_right[t] = q_hi - xi_curr

        hits_left = (r_emp < var_left).astype(int)
        hits_right = (r_emp > var_right).astype(int)

        # Apply exclusion window if provided: drop those indices from backtest
        if exclude_window is not None:
            mask = np.ones(T_eff, dtype=bool)
            start, end = exclude_window
            # The GMM index s corresponds to xi[s+m-1] as the current level
            # and xi[s+m] as the predicted next level. Empirical index on xi is s+m.
            # Convert exclude_window on empirical xi index to GMM/backtest index s:
            gs = max(0, start - m)
            ge = min(T_eff, end - m)
            if gs < ge:
                mask[gs:ge] = False
            hits_left = hits_left[mask]
            hits_right = hits_right[mask]

        for tail, hits in [("left", hits_left), ("right", hits_right)]:
            n = len(hits)
            x = int(hits.sum())
            rate = x / n if n > 0 else 0.0
            lr_uc, p_uc = kupiec_lr(hits, alpha)
            lr_ind, p_ind = christoffersen_ind(hits)
            lr_cc = lr_uc + lr_ind
            p_cc = float(1 - chi2.cdf(lr_cc, df=2))
            key = f"alpha={alpha:.2f}_{tail}"
            results[key] = dict(
                n=n, exceed=x, rate=rate, expected_rate=alpha,
                lr_uc=lr_uc, p_uc=p_uc,
                lr_ind=lr_ind, p_ind=p_ind,
                lr_cc=lr_cc, p_cc=p_cc,
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--variant", choices=sorted(CKPT_PATTERNS.keys()), default="v2")
    ap.add_argument("--out-suffix", default=None)
    ap.add_argument("--exclude-window", default=None,
                    help="Format: market:start:end (empirical xi indices), "
                         "e.g. 'wti:1072:1090' to drop Cushing Apr 2020.")
    args = ap.parse_args()

    exclude_map = {}
    if args.exclude_window:
        mk, s, e = args.exclude_window.split(":")
        exclude_map[mk] = (int(s), int(e))

    suffix = args.out_suffix if args.out_suffix is not None else (
        "" if args.variant == "v2" else f"_{args.variant}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines = []
    lines.append("=" * 110)
    lines.append(f"  VaR BACKTEST - MDN conditional quantiles, 1-day-ahead  (variant: {args.variant})")
    lines.append("=" * 110)
    lines.append(
        f"  {'market':<6} {'level':<6} {'tail':<6} {'N':>5} {'exc':>5} {'rate':>7} {'exp':>6} "
        f"{'LR_uc':>8} {'p_uc':>7} {'LR_ind':>8} {'p_ind':>7} {'LR_cc':>8} {'p_cc':>7}"
    )

    for market in args.markets.split(","):
        res = run_market(market, device, variant=args.variant,
                         exclude_window=exclude_map.get(market))
        if not res:
            print(f"[{market}] SKIP")
            continue
        for key, r in res.items():
            level_str, tail = key.split("_")
            lines.append(
                f"  {market:<6} {level_str:<6} {tail:<6} "
                f"{r['n']:>5d} {r['exceed']:>5d} {r['rate']:>7.4f} {r['expected_rate']:>6.2f} "
                f"{r['lr_uc']:>8.3f} {r['p_uc']:>7.4f} "
                f"{r['lr_ind']:>8.3f} {r['p_ind']:>7.4f} "
                f"{r['lr_cc']:>8.3f} {r['p_cc']:>7.4f}"
            )

    lines.append("")
    lines.append("Interpretation (both p-values > 0.05 => test NOT rejected => correct coverage/indep):")
    lines.append("  Kupiec (LR_uc)       : exceedance rate matches expected level alpha")
    lines.append("  Christoffersen IND   : exceedances are independent (no clustering)")
    lines.append("  Conditional coverage : joint test (LR_cc) with 2 d.o.f.")
    lines.append("=" * 110)

    text = "\n".join(lines)
    print(text)
    out = DATA / f"var_backtest_summary{suffix}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[summary: {out.name}]")


if __name__ == "__main__":
    main()
