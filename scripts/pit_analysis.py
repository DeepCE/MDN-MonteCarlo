#!/usr/bin/env python3
"""Probability Integral Transform (PIT) analysis of the conditional GMM.

For each date t >= m, compute:
    u_t = F_{GMM(t)}(xi_{t+1}^empirical)
where F_{GMM(t)} is the CDF of the one-day-ahead GMM emitted by the MDN given
context x_t = xi[t-m+1..t].  Under the null that the conditional distribution
is correctly specified, {u_t} is i.i.d. Uniform[0,1].

Tests applied:
  (1) Kolmogorov-Smirnov of uniformity on {u_t}
  (2) Anderson-Darling of uniformity (more sensitive to tails)
  (3) Ljung-Box on {u_t - 0.5} for serial independence (lag 10)
  (4) Berkowitz LR on z_t = Phi^{-1}(u_t):
        H0: z_t ~ iid N(0,1)  via AR(1) fit z_t = mu + rho*z_{t-1} + e_t,
        e_t ~ N(0, sigma^2); LR test of (mu=0, rho=0, sigma=1).

Output:
  data/pit_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as spstats
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


def gmm_cdf_at(pi, mu, sigma, x):
    """Evaluate mixture CDF at scalar x for single sample (pi, mu, sigma)."""
    return float(np.sum(pi * norm.cdf(x, loc=mu, scale=sigma)))


def ljung_box(x, lag=10):
    """Manual Ljung-Box on centered series."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    g0 = np.sum(x * x) / n
    q = 0.0
    for k in range(1, lag + 1):
        gk = np.sum(x[:-k] * x[k:]) / n
        q += (gk / g0) ** 2 / (n - k)
    q *= n * (n + 2)
    p = float(1 - chi2.cdf(q, df=lag))
    return float(q), p


def berkowitz_lr(u):
    """Berkowitz LR: z = Phi^{-1}(u); fit AR(1) z_t = mu + rho z_{t-1} + e;
    test H0: (mu=0, rho=0, sigma=1) via LR vs unrestricted AR(1) with free
    (mu, rho, sigma). chi2(3).
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    z = norm.ppf(u)
    n = len(z)
    # Null: z ~ iid N(0,1) => loglik = sum log phi(z)
    ll_null = np.sum(norm.logpdf(z, loc=0, scale=1))
    # Alt: AR(1) with free (mu, rho, sigma^2)
    # OLS on z_t = a + rho z_{t-1} + e
    X = np.column_stack([np.ones(n - 1), z[:-1]])
    y = z[1:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, rho = beta
    resid = y - X @ beta
    sigma2 = np.mean(resid ** 2)
    ll_alt = np.sum(norm.logpdf(y, loc=X @ beta, scale=np.sqrt(sigma2)))
    # Also account for the first obs under null vs alt
    ll_alt += norm.logpdf(z[0], loc=a / (1 - rho + 1e-9),
                          scale=np.sqrt(sigma2 / max(1 - rho ** 2, 1e-6)))
    lr = -2 * (ll_null - ll_alt)
    p = float(1 - chi2.cdf(lr, df=3))
    mu_hat = a
    return float(lr), p, float(mu_hat), float(rho), float(np.sqrt(sigma2))


def run_market(market, device):
    ckpt_path = MODELS / f"mdn_v2_{market}.pt"
    if not ckpt_path.exists():
        print(f"[{market}] SKIP (no checkpoint)")
        return None
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    m = cfg["lookback"]

    prices = load_dat(DATA / COMMODITY_FILES[market])
    xi = loess_detrend(prices)

    T_eff = len(xi) - m
    X = np.stack([xi[t:t + m] for t in range(T_eff)], axis=0).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)

    print(f"[{market}] forward: T_eff={T_eff}, m={m}")
    with torch.no_grad():
        pis, mus, sigmas = [], [], []
        for s in range(0, T_eff, 512):
            p, mu, sg = model(X_t[s:s + 512])
            pis.append(p.cpu().numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(sg.cpu().numpy())
    PI = np.concatenate(pis, axis=0)
    MU = np.concatenate(mus, axis=0)
    SIGMA = np.concatenate(sigmas, axis=0)

    # PIT values: u_t = F_{GMM(t)}(xi[t+m])
    target = xi[m:]  # xi_{t+1} where t runs 0..T_eff-1
    u = np.empty(T_eff)
    for t in range(T_eff):
        u[t] = gmm_cdf_at(PI[t], MU[t], SIGMA[t], float(target[t]))

    # Tests
    ks_stat, ks_p = spstats.kstest(u, "uniform")
    # Anderson-Darling for uniformity via transformation
    # scipy's anderson supports "norm" etc.; for uniform, test z = Phi^{-1}(u)
    # against N(0,1) which is the Berkowitz z transform.
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    ad = spstats.anderson(z, dist="norm")
    ad_stat = float(ad.statistic)
    ad_crit = ad.critical_values  # [15%, 10%, 5%, 2.5%, 1%]
    # Compute approximate p-value via interpolation (fallback: below/above levels)
    if ad_stat < ad_crit[0]:
        ad_p = "> 0.15"
    elif ad_stat >= ad_crit[-1]:
        ad_p = "< 0.01"
    else:
        # linear interp on ln(p)
        levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01])
        ln_p = np.interp(ad_stat, ad_crit, np.log(levels))
        ad_p = f"~{np.exp(ln_p):.3f}"

    # Ljung-Box on u_t (centered)
    lb_stat, lb_p = ljung_box(u - 0.5, lag=10)
    # Berkowitz LR
    bk_stat, bk_p, bk_mu, bk_rho, bk_sigma = berkowitz_lr(u)

    return {
        "n": T_eff,
        "u_mean": float(u.mean()),
        "u_std": float(u.std(ddof=1)),
        "ks": (ks_stat, ks_p),
        "ad": (ad_stat, ad_p),
        "lb": (lb_stat, lb_p),
        "bk": (bk_stat, bk_p, bk_mu, bk_rho, bk_sigma),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines = []
    lines.append("=" * 100)
    lines.append(
        "  PIT analysis: u_t = F_{GMM(t)}(xi_{t+1}^emp), t >= m. "
        "Under correct conditional specification, u_t ~ iid U[0,1]."
    )
    lines.append("=" * 100)
    lines.append(
        f"  {'market':<6} {'N':>5} {'u_mean':>7} {'u_std':>7} "
        f"{'KS':>7} {'p_KS':>8} {'AD':>7} {'p_AD':>10} "
        f"{'LB(10)':>8} {'p_LB':>8} {'Berk':>7} {'p_Berk':>8}"
    )
    lines.append("  " + "-" * 98)

    for market in ("psv", "pun", "pjm", "wti"):
        r = run_market(market, device)
        if r is None:
            continue
        ks_s, ks_p = r["ks"]
        ad_s, ad_p = r["ad"]
        lb_s, lb_p = r["lb"]
        bk_s, bk_p, bk_mu, bk_rho, bk_sig = r["bk"]
        lines.append(
            f"  {market:<6} {r['n']:>5d} {r['u_mean']:>7.4f} {r['u_std']:>7.4f} "
            f"{ks_s:>7.4f} {ks_p:>8.4f} {ad_s:>7.3f} {str(ad_p):>10} "
            f"{lb_s:>8.2f} {lb_p:>8.4f} {bk_s:>7.2f} {bk_p:>8.4f}"
        )
        lines.append(
            f"         Berkowitz fit: mu={bk_mu:+.3f}  rho={bk_rho:+.3f}  "
            f"sigma={bk_sig:.3f} (H0: mu=0, rho=0, sigma=1)"
        )

    lines.append("")
    lines.append("Legend: U[0,1] has mean=0.5, std=sqrt(1/12)=0.2887.")
    lines.append(
        "  KS (Kolmogorov-Smirnov): p>0.05 => uniformity not rejected."
    )
    lines.append(
        "  AD (Anderson-Darling of z=Phi^{-1}(u) vs N(0,1)): tail-sensitive; "
        "p>0.05 => normality not rejected."
    )
    lines.append(
        "  LB(10) (Ljung-Box on u_t - 0.5, 10 lags): p>0.05 => serial "
        "independence not rejected."
    )
    lines.append(
        "  Berkowitz LR (3 d.o.f.): p>0.05 => joint (mu=0, rho=0, sigma=1) "
        "not rejected."
    )
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    out = DATA / "pit_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
