#!/usr/bin/env python3
"""LOESS sensitivity analysis for the 4 energy markets.

Risponde a R1 punto 4.1: verifica che la scelta del smoothing fraction (frac=0.10
usato nel paper) non influisce sostanzialmente sulle statistiche descrittive della
detrended series xi_t e dei log-returns r_t.

Confronta frac in {0.05, 0.10, 0.15, 0.20} per ciascun mercato.

Output: data/loess_sensitivity.txt con tabella comparativa.
"""

from pathlib import Path
import numpy as np


DATA = Path(__file__).resolve().parent.parent / "data"
FRACS = [0.05, 0.10, 0.15, 0.20]

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


def loess_detrend(prices: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
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
    return trend, log_prices - trend


def moments(x: np.ndarray) -> dict:
    m, s = float(np.mean(x)), float(np.std(x))
    if s == 0:
        return dict(std=0.0, skew=0.0, kurt=0.0)
    z = (x - m) / s
    return dict(std=s, skew=float(np.mean(z ** 3)), kurt=float(np.mean(z ** 4)))


def window_days(frac: float, n: int) -> int:
    return max(5, int(n * frac) | 1)


def main():
    lines = []
    lines.append("=" * 102)
    lines.append("  LOESS SENSITIVITY ANALYSIS - 4 ENERGY MARKETS")
    lines.append("  Statistics of detrended log-prices xi_t and log-returns r_t across frac in {0.05,0.10,0.15,0.20}")
    lines.append("=" * 102)

    for market_name, filename in MARKETS:
        prices = load_dat(DATA / filename)
        n = len(prices)
        lines.append("")
        lines.append(f"-- {market_name} (N={n})  " + "-" * (90 - len(market_name) - len(str(n))))
        lines.append(
            f"  {'frac':>6} {'window':>8} "
            f"{'xi_std':>10} {'xi_skew':>10} {'xi_kurt':>10}   "
            f"{'r_std':>10} {'r_skew':>10} {'r_kurt':>10}"
        )
        for frac in FRACS:
            _, xi = loess_detrend(prices, frac=frac)
            r = np.diff(xi)
            mx = moments(xi)
            mr = moments(r)
            w = window_days(frac, n)
            marker = " <-- paper" if frac == 0.10 else ""
            lines.append(
                f"  {frac:>6.2f} {w:>8d} "
                f"{mx['std']:>10.4f} {mx['skew']:>+10.3f} {mx['kurt']:>10.3f}   "
                f"{mr['std']:>10.4f} {mr['skew']:>+10.3f} {mr['kurt']:>10.3f}{marker}"
            )

    lines.append("")
    lines.append("=" * 102)
    lines.append("Interpretation:")
    lines.append("  - frac=0.05: windows ~5% of sample (~90 days for N=1826); trend follows short-term oscillations.")
    lines.append("  - frac=0.10: baseline used in paper (~180 days); separates low-freq trend from medium-freq cycles.")
    lines.append("  - frac=0.15: windows ~15% (~275 days); smoother trend.")
    lines.append("  - frac=0.20: windows ~20% (~365 days); trend close to annual average.")
    lines.append("  - r_t statistics (std, skew, kurt) should remain close across frac if choice is not critical.")
    lines.append("=" * 102)

    text = "\n".join(lines)
    print(text)
    out = DATA / "loess_sensitivity.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[written to {out.name}]")


if __name__ == "__main__":
    main()
