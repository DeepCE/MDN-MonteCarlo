#!/usr/bin/env python3
"""Benchmark parametric models: OU + AR(1)-GARCH(1,1).

Due modelli fittati sulla serie detrended xi_t per ciascun mercato (PSV, PUN,
PJM, WTI), quindi generazione di 5000 traiettorie Monte Carlo per modello.

Scopo: fornire benchmark quantitativi per rispondere ai commenti:
  - R1 #2, #3, #6 (abstract + literature + moment validation)
  - R2 #1 (benchmark probabilistico)

Modelli:
  * OU (discretizzato, Gaussian innovations):
        xi_t = phi * xi_{t-1} + sigma * eps_t,  eps ~ N(0,1)
        alpha = 1 - phi,   tau_{1/2} = ln(2) / alpha
  * AR(1)-GARCH(1,1) (Gaussian standardized innovations):
        xi_t = mu + phi*(xi_{t-1} - mu) + eps_t
        eps_t = sigma_t * z_t,   z_t ~ N(0,1)
        sigma_t^2 = omega + g1*eps_{t-1}^2 + g2*sigma_{t-1}^2

Output:
  data/benchmarks/<market>_ou_paths.npz
  data/benchmarks/<market>_garch_paths.npz
  data/benchmarks/fit_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from arch import arch_model


# ---------- config ----------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BENCH = DATA / "benchmarks"
BENCH.mkdir(parents=True, exist_ok=True)

N_TRAJ = 5000
LOESS_FRAC = 0.10
# Seed master: use per-market seed so diff markets are independent but reproducible
MASTER_SEED = 42

MARKETS = [
    ("psv", "PSV gas",   "gas_1826.dat"),
    ("pun", "PUN power", "power_1826.dat"),
    ("pjm", "PJM power", "pjm_2451.dat"),
    ("wti", "WTI oil",   "wti_2501.dat"),
]


# ---------- data ----------

def load_prices(path: Path) -> np.ndarray:
    prices = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prices.append(float(line.replace(",", ".")))
    return np.array(prices)


def loess_detrend(prices: np.ndarray, frac: float = LOESS_FRAC) -> np.ndarray:
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


# ---------- OU fit / simulate ----------

def fit_ou(xi: np.ndarray) -> dict:
    """OLS fit xi_t = phi * xi_{t-1} + sigma * eps_t (no intercept: xi is ~ centered).

    Restituisce phi, sigma, alpha=1-phi, tau_half, SE(phi), SE(alpha).
    """
    x = xi[:-1]
    y = xi[1:]
    phi = float(np.sum(x * y) / np.sum(x * x))
    resid = y - phi * x
    n = len(x)
    sigma2 = float(np.sum(resid ** 2) / (n - 1))
    sigma = float(np.sqrt(sigma2))
    # SE(phi) under homoskedastic OLS: sigma / sqrt(sum x^2)
    se_phi = float(sigma / np.sqrt(np.sum(x * x)))
    alpha = 1.0 - phi
    se_alpha = se_phi  # derivative 1:1
    tau_half = float(np.log(2) / alpha) if alpha > 0 else float("inf")
    return dict(phi=phi, sigma=sigma, alpha=alpha, tau_half=tau_half,
                se_phi=se_phi, se_alpha=se_alpha, n=n)


def simulate_ou(phi: float, sigma: float, xi0: float, n_steps: int,
                n_traj: int, rng: np.random.Generator) -> np.ndarray:
    """Genera n_traj traiettorie OU di lunghezza n_steps partendo da xi0."""
    paths = np.empty((n_traj, n_steps), dtype=np.float64)
    x = np.full(n_traj, xi0, dtype=np.float64)
    for t in range(n_steps):
        eps = rng.standard_normal(n_traj)
        x = phi * x + sigma * eps
        paths[:, t] = x
    return paths


# ---------- AR(1)-GARCH(1,1) fit / simulate ----------

def fit_ar_garch(xi: np.ndarray) -> dict:
    """AR(1) mean + GARCH(1,1) variance (Gaussian innovations)."""
    # arch expects series; we fit on xi directly (levels, weakly stationary)
    model = arch_model(xi, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal",
                       rescale=False)
    res = model.fit(disp="off", show_warning=False)
    params = res.params
    # arch uses: Const (mu0), xi[1] (phi), omega, alpha[1] (g1), beta[1] (g2)
    mu0 = float(params["Const"])
    phi = float(params.get("xi[1]", params.get(f"{model.y.name}[1]" if hasattr(model.y, 'name') else "y[1]", None)))
    # robust access: params index can be 'Const','xi[1]','omega','alpha[1]','beta[1]'
    phi = float(params.iloc[1])
    omega = float(params["omega"])
    g1 = float(params["alpha[1]"])
    g2 = float(params["beta[1]"])
    # Stationarity check and unconditional mean of xi
    mu_uncond = mu0 / (1.0 - phi) if abs(phi) < 1 else float("nan")
    return dict(mu0=mu0, phi=phi, mu_uncond=mu_uncond, omega=omega, g1=g1, g2=g2,
                loglik=float(res.loglikelihood), aic=float(res.aic), bic=float(res.bic),
                n=len(xi))


def simulate_ar_garch(fit: dict, xi0: float, n_steps: int, n_traj: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Simula AR(1)-GARCH(1,1) in forma ricorsiva."""
    mu0 = fit["mu0"]
    phi = fit["phi"]
    omega = fit["omega"]
    g1 = fit["g1"]
    g2 = fit["g2"]

    paths = np.empty((n_traj, n_steps), dtype=np.float64)
    x = np.full(n_traj, xi0, dtype=np.float64)
    # Initialize sigma^2 at unconditional variance of epsilon (= omega / (1 - g1 - g2)
    # if g1+g2<1, else omega)
    denom = 1.0 - g1 - g2
    sigma2 = np.full(n_traj, omega / denom if denom > 0 else omega, dtype=np.float64)
    eps_prev_sq = sigma2.copy()  # init eps^2 at uncond variance

    for t in range(n_steps):
        # update conditional variance from previous shock
        sigma2 = omega + g1 * eps_prev_sq + g2 * sigma2
        z = rng.standard_normal(n_traj)
        eps = np.sqrt(sigma2) * z
        x = mu0 + phi * x + eps
        paths[:, t] = x
        eps_prev_sq = eps ** 2
    return paths


# ---------- main ----------

def moments(x: np.ndarray) -> dict:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s == 0:
        return dict(mean=m, std=0.0, skew=0.0, kurt=0.0)
    z = (x - m) / s
    return dict(mean=m, std=s, skew=float(np.mean(z ** 3)), kurt=float(np.mean(z ** 4)))


def main():
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("  BENCHMARK PARAMETRIC FITS + 5000-PATH GENERATION")
    summary_lines.append("=" * 90)

    for key, market, filename in MARKETS:
        print(f"\n[{market}] loading {filename} ...")
        prices = load_prices(DATA / filename)
        xi = loess_detrend(prices)
        n_steps = len(xi)  # generiamo path completi per poi confrontare

        rng = np.random.default_rng(MASTER_SEED + hash(key) % 10_000)

        # OU
        print(f"[{market}] fit OU ...")
        ou = fit_ou(xi)
        print(f"  phi={ou['phi']:.6f} (SE {ou['se_phi']:.6f})  "
              f"sigma={ou['sigma']:.6f}  alpha={ou['alpha']:.6f}  "
              f"tau_half={ou['tau_half']:.2f} d")
        print(f"[{market}] simulate OU: {N_TRAJ} paths x {n_steps} steps ...")
        ou_paths = simulate_ou(ou["phi"], ou["sigma"], xi[0], n_steps, N_TRAJ, rng)

        # AR-GARCH
        print(f"[{market}] fit AR(1)-GARCH(1,1) ...")
        gh = fit_ar_garch(xi)
        print(f"  mu0={gh['mu0']:.6f}  phi={gh['phi']:.6f}  "
              f"omega={gh['omega']:.6e}  g1={gh['g1']:.4f}  g2={gh['g2']:.4f}  "
              f"loglik={gh['loglik']:.2f}")
        print(f"[{market}] simulate GARCH: {N_TRAJ} paths x {n_steps} steps ...")
        gh_paths = simulate_ar_garch(gh, xi[0], n_steps, N_TRAJ, rng)

        # Save paths
        ou_out = BENCH / f"{key}_ou_paths.npz"
        gh_out = BENCH / f"{key}_garch_paths.npz"
        np.savez_compressed(ou_out, paths=ou_paths, fit=np.array([ou], dtype=object))
        np.savez_compressed(gh_out, paths=gh_paths, fit=np.array([gh], dtype=object))
        print(f"[{market}] saved: {ou_out.name} ({ou_paths.nbytes/1e6:.1f} MB), "
              f"{gh_out.name} ({gh_paths.nbytes/1e6:.1f} MB)")

        # Summary
        xi_mom = moments(xi)
        r_emp = np.diff(xi)
        r_emp_mom = moments(r_emp)
        r_ou_all = np.diff(ou_paths, axis=1).ravel()
        r_gh_all = np.diff(gh_paths, axis=1).ravel()
        r_ou_mom = moments(r_ou_all)
        r_gh_mom = moments(r_gh_all)

        summary_lines.append("")
        summary_lines.append(f"-- {market} (N={len(xi)}) " + "-" * (76 - len(market) - len(str(len(xi)))))
        summary_lines.append(
            f"  OU fit:    phi={ou['phi']:.5f} (SE {ou['se_phi']:.5f})  "
            f"sigma={ou['sigma']:.5f}  alpha={ou['alpha']:.5f}  tau_half={ou['tau_half']:.2f} d"
        )
        summary_lines.append(
            f"  GARCH fit: mu0={gh['mu0']:+.6f}  phi={gh['phi']:.5f}  "
            f"omega={gh['omega']:.4e}  g1={gh['g1']:.4f}  g2={gh['g2']:.4f}  "
            f"sum(g1+g2)={gh['g1']+gh['g2']:.4f}  loglik={gh['loglik']:.2f}"
        )
        summary_lines.append(
            f"  r_t emp : std={r_emp_mom['std']:.4f}  skew={r_emp_mom['skew']:+.3f}  kurt={r_emp_mom['kurt']:.2f}"
        )
        summary_lines.append(
            f"  r_t OU  : std={r_ou_mom['std']:.4f}  skew={r_ou_mom['skew']:+.3f}  kurt={r_ou_mom['kurt']:.2f}"
        )
        summary_lines.append(
            f"  r_t GARCH: std={r_gh_mom['std']:.4f}  skew={r_gh_mom['skew']:+.3f}  kurt={r_gh_mom['kurt']:.2f}"
        )

    summary_lines.append("")
    summary_lines.append("=" * 90)
    text = "\n".join(summary_lines)
    print("\n" + text)

    out = BENCH / "fit_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[summary: {out}]")


if __name__ == "__main__":
    main()
