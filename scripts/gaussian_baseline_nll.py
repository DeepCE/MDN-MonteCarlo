#!/usr/bin/env python3
"""Gaussian baseline NLL per ciascun mercato.

Risposta a R1 punto 5.d: fornisce la NLL "naive" sotto l'assunzione che i
log-returns r_t siano i.i.d. N(mu, sigma^2) fittati su tutta la serie.
Questa serve come scala di riferimento contro cui leggere la val NLL dell'MDN.

Improvement = (NLL_gaussian - NLL_mdn) / |NLL_gaussian|    (e` positivo se
l'MDN fa meglio della gaussiana, che e' il caso atteso per log-returns
heavy-tailed).

Output: stampa su stdout + salva in data/gaussian_baseline.txt.
"""

from pathlib import Path
import numpy as np


DATA = Path(__file__).resolve().parent.parent / "data"

MARKETS = [
    ("PSV gas", "gas_1826.dat"),
    ("PUN power", "power_1826.dat"),
    ("PJM power", "pjm_2451.dat"),
    ("WTI oil", "wti_2501.dat"),
]


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


def gaussian_nll(samples: np.ndarray) -> tuple[float, float, float]:
    """Per-sample mean NLL under univariate Gaussian fit to `samples` (MLE)."""
    mu = float(np.mean(samples))
    sigma2 = float(np.var(samples))  # MLE (not unbiased)
    nll_per_sample = 0.5 * (np.log(2 * np.pi * sigma2) + (samples - mu) ** 2 / sigma2)
    return float(np.mean(nll_per_sample)), mu, np.sqrt(sigma2)


def main():
    lines = []
    lines.append("=" * 80)
    lines.append("  GAUSSIAN BASELINE NLL (unconditional i.i.d. Gaussian fit on r_t)")
    lines.append("=" * 80)
    lines.append(
        f"  {'Market':<14} {'N_returns':>10} {'mu':>12} {'sigma':>12} {'NLL_gauss':>12}"
    )

    result_records = {}
    for market, filename in MARKETS:
        prices = load_dat(DATA / filename)
        xi = loess_detrend(prices, frac=0.1)
        r = np.diff(xi)
        nll, mu, sigma = gaussian_nll(r)
        lines.append(
            f"  {market:<14} {len(r):>10d} {mu:>+12.6f} {sigma:>12.6f} {nll:>12.4f}"
        )
        result_records[market] = dict(N=len(r), mu=mu, sigma=sigma, nll=nll)

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  NLL_gauss is the average per-sample negative log-likelihood under")
    lines.append("  the best i.i.d. Gaussian fit to the detrended log-returns r_t.")
    lines.append("  The MDN validation NLL can be compared against this baseline:")
    lines.append("    improvement = NLL_gauss - NLL_mdn   (positive => MDN beats Gaussian)")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)

    out = DATA / "gaussian_baseline.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[written to {out.name}]")


if __name__ == "__main__":
    main()
