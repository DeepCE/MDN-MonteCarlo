#!/usr/bin/env python3
"""Pillar 2 quantitativo: IQR condizionale della GMM MDN nel tempo + KL divergence
dalla GMM "media" della serie + confronto con volatility empirica.

Obiettivo (risposta R1 punto 8c + R2 punto 1):
  Dimostrare QUANTITATIVAMENTE che la shape condizionale della GMM varia
  in funzione della storia recente, con magnitudine che correla con la volatilita'
  realizzata. I modelli parametrici (GARCH incluso) hanno shape relativa costante:
  varia la scala ma la distribuzione normalizzata e' fissa.

Per ciascun mercato:
  1. Carica modello MDN finale.
  2. Per ogni timestep t, calcola GMM condizionale p(xi_{t+1} | x_t).
  3. Calcola:
       a. IQR condizionale (quantile-based spread)
       b. p-th quantile spread (range Q95-Q5) per tail-focused metric
       c. KL divergence dalla GMM "media" unconditional (pooled mixture)
  4. Serie temporale IQR_t confrontata con realized_vol_t (rolling 30d std r_t)
  5. Calcolo coefficiente di correlazione Pearson IQR_t vs realized_vol_t

Output:
  data/pillar2/<market>_pillar2.npz  (series temporali)
  data/pillar2_summary.txt           (correlazioni + interpretazione)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdn_models import load_checkpoint_model  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = DATA / "pillar2"
OUT.mkdir(parents=True, exist_ok=True)

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


# Model classes imported via mdn_models.load_checkpoint_model


def gmm_quantiles_batch(PI: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray,
                         qs: np.ndarray, grid_size: int = 1024) -> np.ndarray:
    """Batched quantiles: per ogni t, calcola quantili qs dalla GMM (pi,mu,sigma)_t.

    PI,MU,SIGMA: shape (T, K).  qs: shape (Q,).
    Ritorna: shape (T, Q).

    Per efficienza usa un grid per mercato (fisso sul range) e calcola
    cdf_t(x) = sum_k pi_{tk} * Phi((x-mu_{tk})/sigma_{tk}) vettorizzato.
    """
    T, K = PI.shape
    # Range globale
    lo = float((MU - 6 * SIGMA).min())
    hi = float((MU + 6 * SIGMA).max())
    xs = np.linspace(lo, hi, grid_size)  # (G,)
    # Calcolo cdf_t per tutti t vettorizzato
    # cdf[t, g] = sum_k pi[t,k] * Phi((xs[g] - mu[t,k]) / sigma[t,k])
    # shape manipolazione: (T, K, G)
    xs_b = xs[None, None, :]           # (1, 1, G)
    mu_b = MU[:, :, None]               # (T, K, 1)
    sigma_b = SIGMA[:, :, None]         # (T, K, 1)
    z = (xs_b - mu_b) / sigma_b         # (T, K, G)
    phi = norm.cdf(z)                   # (T, K, G)
    cdf = (PI[:, :, None] * phi).sum(axis=1)  # (T, G)
    # Per ogni t e q, searchsorted
    out = np.empty((T, len(qs)), dtype=np.float64)
    for i, q in enumerate(qs):
        # per ogni riga di cdf trova idx con cdf[idx] >= q
        idx = np.array([np.searchsorted(cdf[t], q) for t in range(T)])
        idx = np.clip(idx, 1, grid_size - 1)
        x0 = xs[idx - 1]
        x1 = xs[idx]
        y0 = cdf[np.arange(T), idx - 1]
        y1 = cdf[np.arange(T), idx]
        denom = np.where(y1 != y0, y1 - y0, 1.0)
        out[:, i] = x0 + (q - y0) / denom * (x1 - x0)
    return out


def gmm_moments_batch(PI: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Batched mean and std of per-timestep mixtures. Shapes (T, K) -> (T,)."""
    m = (PI * MU).sum(axis=1)
    var = (PI * (SIGMA ** 2 + MU ** 2)).sum(axis=1) - m ** 2
    var = np.clip(var, 1e-30, None)
    return m, np.sqrt(var)


def kl_mc(pi1, mu1, sigma1, pi2, mu2, sigma2, n_samples: int = 10_000,
          rng: np.random.Generator | None = None) -> float:
    """KL(p1 || p2) for two Gaussian mixtures via Monte Carlo."""
    if rng is None:
        rng = np.random.default_rng(0)
    # sample from p1
    comp = rng.choice(len(pi1), size=n_samples, p=pi1 / pi1.sum())
    z = rng.standard_normal(n_samples)
    samples = mu1[comp] + sigma1[comp] * z
    # log density of p1 and p2 on samples
    def logpdf(x, pi, mu, sigma):
        # x: (N,), pi,mu,sigma: (K,)
        # log sum_k pi_k * N(x; mu_k, sigma_k^2)
        log_pi = np.log(pi + 1e-20)
        log_norm = -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-20)
        diff = (x[:, None] - mu[None, :]) / (sigma[None, :] + 1e-20)
        log_g = log_norm[None, :] - 0.5 * diff ** 2
        return np.logaddexp.reduce(log_pi[None, :] + log_g, axis=1)
    lp1 = logpdf(samples, pi1, mu1, sigma1)
    lp2 = logpdf(samples, pi2, mu2, sigma2)
    return float(np.mean(lp1 - lp2))


def pooled_gmm(all_pi: np.ndarray, all_mu: np.ndarray, all_sigma: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatena K*T components in un'unica mixture con pesi (1/T)*pi_{jt}."""
    T, K = all_pi.shape
    flat_pi = (all_pi / T).ravel()
    flat_mu = all_mu.ravel()
    flat_sigma = all_sigma.ravel()
    return flat_pi, flat_mu, flat_sigma


CKPT_PATTERNS = {
    "retrained": "mdn_final_{market}.pt",
    "v2": "mdn_v2_{market}.pt",
}


def run_market(market: str, device: torch.device, variant: str = "v2") -> dict:
    pattern = CKPT_PATTERNS[variant]
    ckpt_path = MODELS / pattern.format(market=market)
    if not ckpt_path.exists():
        print(f"[{market}] SKIP (no checkpoint at {ckpt_path})")
        return {}
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    m = cfg["lookback"]
    K = cfg["n_components"]

    prices = load_dat(DATA / COMMODITY_FILES[market])
    xi = loess_detrend(prices)

    # Costruisci tutte le finestre x_t = xi[t-m:t] per t = m..N-1
    T_eff = len(xi) - m
    X = np.stack([xi[t:t + m] for t in range(T_eff)], axis=0).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)

    print(f"[{market}] forward pass: {T_eff} timesteps, m={m}, K={K} ...")
    with torch.no_grad():
        pi_all, mu_all, sigma_all = [], [], []
        batch = 512
        for s in range(0, T_eff, batch):
            pi, mu, sigma = model(X_t[s:s + batch])
            pi_all.append(pi.cpu().numpy())
            mu_all.append(mu.cpu().numpy())
            sigma_all.append(sigma.cpu().numpy())
    PI = np.concatenate(pi_all, axis=0)       # (T_eff, K)
    MU = np.concatenate(mu_all, axis=0)
    SIGMA = np.concatenate(sigma_all, axis=0)

    # GMM "di riferimento": pooled mixture su tutti i timestep
    ref_pi, ref_mu, ref_sigma = pooled_gmm(PI, MU, SIGMA)

    print(f"[{market}] computing IQR/range95 batched ...", flush=True)
    quantiles = gmm_quantiles_batch(PI, MU, SIGMA,
                                    np.array([0.05, 0.25, 0.75, 0.95]),
                                    grid_size=1024)
    iqr_t = quantiles[:, 2] - quantiles[:, 1]      # q75 - q25
    range95_t = quantiles[:, 3] - quantiles[:, 0]  # q95 - q05

    print(f"[{market}] computing mean/std batched ...", flush=True)
    mean_t, std_t = gmm_moments_batch(PI, MU, SIGMA)

    # KL MC: sotto-campioniamo ogni timestep (solo ogni STRIDE-th)
    # per ridurre il costo mantenendo una serie temporale riconoscibile.
    # Strategia: calcolare KL solo su ~500 timestep equispaziati.
    STRIDE = max(1, T_eff // 500)
    t_sample = np.arange(0, T_eff, STRIDE)
    rng = np.random.default_rng(42)
    print(f"[{market}] KL MC on {len(t_sample)} sampled timesteps (stride={STRIDE}) ...",
          flush=True)
    kl_sampled = np.empty(len(t_sample))
    for i, t in enumerate(t_sample):
        kl_sampled[i] = kl_mc(PI[t], MU[t], SIGMA[t],
                               ref_pi, ref_mu, ref_sigma,
                               n_samples=500, rng=rng)
    # Interpola agli altri timestep (lineare)
    kl_t = np.interp(np.arange(T_eff), t_sample, kl_sampled)

    # Realized volatility empirica (rolling 30-day std di r_t)
    r_emp = np.diff(xi)
    win = 30
    rv_rolling = np.array([
        float(r_emp[max(0, i - win):i + 1].std()) if i >= 1 else 0.0
        for i in range(len(r_emp))
    ])
    # Align: iqr_t is at time t (predicting t+1), so compare with rv_rolling[t-1]
    rv_aligned = rv_rolling[m - 1:m - 1 + T_eff]  # shift to match IQR index
    if len(rv_aligned) > T_eff:
        rv_aligned = rv_aligned[:T_eff]
    elif len(rv_aligned) < T_eff:
        rv_aligned = np.pad(rv_aligned, (0, T_eff - len(rv_aligned)), mode='edge')

    # Correlazioni
    corr_iqr_rv = float(np.corrcoef(iqr_t, rv_aligned)[0, 1])
    corr_r95_rv = float(np.corrcoef(range95_t, rv_aligned)[0, 1])
    corr_std_rv = float(np.corrcoef(std_t, rv_aligned)[0, 1])
    # std/iqr ratio variation: se shape fosse costante (GARCH-like), std_t/iqr_t sarebbe costante
    shape_ratio = std_t / np.where(iqr_t > 0, iqr_t, 1.0)
    cv_shape = float(shape_ratio.std() / shape_ratio.mean()) if shape_ratio.mean() > 0 else float("nan")

    summary = dict(
        T_eff=T_eff, m=m, K=K,
        corr_iqr_rv=corr_iqr_rv,
        corr_r95_rv=corr_r95_rv,
        corr_std_rv=corr_std_rv,
        kl_mean=float(kl_t.mean()),
        kl_max=float(kl_t.max()),
        kl_p95=float(np.percentile(kl_t, 95)),
        shape_ratio_mean=float(shape_ratio.mean()),
        shape_ratio_cv=cv_shape,
    )

    # Save
    out = OUT / f"{market}_pillar2.npz"
    np.savez_compressed(
        out,
        iqr_t=iqr_t, range95_t=range95_t, mean_t=mean_t, std_t=std_t, kl_t=kl_t,
        rv_rolling=rv_aligned, r_emp=r_emp, xi=xi,
        summary=np.array([summary], dtype=object),
    )
    print(f"[{market}] saved {out.name}  corr(IQR,RV)={corr_iqr_rv:+.3f}  "
          f"KL_mean={summary['kl_mean']:.3f}  shape_ratio_cv={cv_shape:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--variant", choices=sorted(CKPT_PATTERNS.keys()), default="v2",
                    help="which checkpoint pattern to load")
    ap.add_argument("--out-suffix", default=None,
                    help="suffix for summary txt (default: '' for v2, '_<variant>' otherwise)")
    args = ap.parse_args()

    suffix = args.out_suffix if args.out_suffix is not None else (
        "" if args.variant == "v2" else f"_{args.variant}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  variant: {args.variant}")

    lines = []
    lines.append("=" * 96)
    lines.append(f"  PILLAR 2 QUANTITATIVE METRICS  (variant: {args.variant})")
    lines.append("  Conditional IQR, range95, KL divergence, shape-ratio CV")
    lines.append("=" * 96)
    lines.append(f"  {'market':<6} {'T_eff':>6} {'corr(IQR,RV)':>14} {'corr(R95,RV)':>14} "
                 f"{'KL_mean':>10} {'KL_p95':>10} {'shape_CV':>10}")

    for market in args.markets.split(","):
        s = run_market(market, device, variant=args.variant)
        if not s:
            continue
        lines.append(
            f"  {market:<6} {s['T_eff']:>6d} "
            f"{s['corr_iqr_rv']:>+14.3f} {s['corr_r95_rv']:>+14.3f} "
            f"{s['kl_mean']:>10.4f} {s['kl_p95']:>10.4f} {s['shape_ratio_cv']:>10.4f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  corr(IQR,RV) :  correlation between MDN conditional IQR and realized 30d vol.")
    lines.append("                  High positive corr => IQR expands with volatility (Pillar 2 scale).")
    lines.append("  shape_CV    :  coefficient of variation of std_t/IQR_t across timesteps.")
    lines.append("                 In a scale-only model (e.g. GARCH with fixed innov dist) this is ~0.")
    lines.append("                 In MDN, CV>0 indicates the *shape* of the conditional law changes,")
    lines.append("                 not only its scale => Pillar 2 is about shape, not just scale.")
    lines.append("  KL_mean     :  average KL divergence from pooled reference mixture.")
    lines.append("                 Higher => more heterogeneous conditional distributions across time.")
    lines.append("=" * 96)

    text = "\n".join(lines)
    print("\n" + text)

    out_txt = DATA / f"pillar2_summary{suffix}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[summary: {out_txt.name}]")


if __name__ == "__main__":
    main()
