#!/usr/bin/env python3
"""Preprocessing dei 4 mercati energetici (PSV, PUN, PJM, WTI).

Converte PJM e WTI da Excel a .dat, gestisce l'evento negativo WTI 2020-04-20
(forward fill), applica LOESS detrending coerente con il codice esistente, e
calcola statistiche descrittive per raw prices, detrended log-prices e log-returns.

Output:
  - data/pjm_{N}.dat   (prezzi one-per-line)
  - data/wti_{N}.dat   (prezzi one-per-line)
  - data/preprocess_summary.txt  (tabella descrittiva)
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent.parent / "data"


def load_dat(path: Path) -> np.ndarray:
    prices = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prices.append(float(line.replace(",", ".")))
    return np.array(prices)


def load_excel_series(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def forward_fill_nonpositive(df: pd.DataFrame, col: str = "price") -> tuple[pd.DataFrame, list]:
    """Sostituisce valori <=0 con il prezzo valido precedente. Ritorna DF + log."""
    df = df.copy()
    fills = []
    for i in range(len(df)):
        if df.loc[i, col] <= 0:
            if i == 0:
                raise ValueError("Non-positive price in first row; cannot forward-fill.")
            prev = df.loc[i - 1, col]
            fills.append((df.loc[i, "date"], df.loc[i, col], prev))
            df.loc[i, col] = prev
    return df, fills


def winsorize_window(df: pd.DataFrame, dates: list, col: str = "price") -> tuple[pd.DataFrame, list]:
    """Sostituisce i prezzi nelle date indicate col valore valido immediatamente precedente.

    Usato per smussare anomalie di settlement prolungate (WTI 2020-04-20..22).
    """
    df = df.copy()
    fills = []
    target_dates = pd.to_datetime(dates)
    for d in target_dates:
        mask = df["date"] == d
        if not mask.any():
            continue
        idx = int(df.index[mask][0])
        if idx == 0:
            raise ValueError("Cannot winsorize first row.")
        prev = df.loc[idx - 1, col]
        fills.append((df.loc[idx, "date"], df.loc[idx, col], prev))
        df.loc[idx, col] = prev
    return df, fills


def write_dat(prices: np.ndarray, path: Path) -> None:
    with open(path, "w") as f:
        for p in prices:
            f.write(f"{p:.4f}\n")


def loess_detrend(prices: np.ndarray, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-kernel LOESS detrending, coerente con scripts/train_mdn_final.py."""
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
    detrended = log_prices - trend
    return log_prices, trend, detrended


def moments(x: np.ndarray) -> dict:
    m, s = float(np.mean(x)), float(np.std(x))
    if s == 0:
        return dict(mean=m, std=0.0, skew=0.0, kurt=0.0, min=float(x.min()), max=float(x.max()))
    z = (x - m) / s
    return dict(
        mean=m,
        std=s,
        skew=float(np.mean(z ** 3)),
        kurt=float(np.mean(z ** 4)),
        min=float(x.min()),
        max=float(x.max()),
    )


def describe_market(name: str, prices: np.ndarray, frac: float = 0.1) -> dict:
    log_p, trend, xi = loess_detrend(prices, frac=frac)
    returns = np.diff(xi)
    return {
        "name": name,
        "n": len(prices),
        "raw": moments(prices),
        "detrended": moments(xi),
        "returns": moments(returns),
        "log_mean": float(np.mean(log_p)),
        "trend_range": (float(trend.min()), float(trend.max())),
    }


def fmt_row(label: str, stats: dict, fmt: str = "{:.4f}") -> str:
    return (
        f"  {label:<14} "
        f"{fmt.format(stats['mean']):>12} "
        f"{fmt.format(stats['std']):>12} "
        f"{fmt.format(stats['min']):>12} "
        f"{fmt.format(stats['max']):>12} "
        f"{stats['skew']:>+8.3f} "
        f"{stats['kurt']:>8.3f}"
    )


def main():
    # ---------- PJM ----------
    pjm_df = load_excel_series(DATA / "pjm_10yr.xlsx")
    pjm_prices = pjm_df["price"].to_numpy(dtype=float)
    n_pjm = len(pjm_prices)
    pjm_out = DATA / f"pjm_{n_pjm}.dat"
    write_dat(pjm_prices, pjm_out)
    print(f"[PJM] {n_pjm} obs, {pjm_df['date'].iloc[0].date()} -> {pjm_df['date'].iloc[-1].date()}")
    print(f"      written to {pjm_out.name}")

    # ---------- WTI ----------
    wti_df = load_excel_series(DATA / "wti_10yr.xls")
    # Step 1: forward-fill non-positive (2020-04-20 = -36.98)
    wti_df_ff, fills_neg = forward_fill_nonpositive(wti_df)
    # Step 2: winsorize 3-day Cushing storage anomaly window (20, 21, 22 April 2020)
    anomaly_dates = ["2020-04-20", "2020-04-21", "2020-04-22"]
    wti_df_clean, fills_wind = winsorize_window(wti_df_ff, anomaly_dates)
    wti_prices = wti_df_clean["price"].to_numpy(dtype=float)
    n_wti = len(wti_prices)
    wti_out = DATA / f"wti_{n_wti}.dat"
    write_dat(wti_prices, wti_out)
    print(f"[WTI] {n_wti} obs, {wti_df['date'].iloc[0].date()} -> {wti_df['date'].iloc[-1].date()}")
    print(f"      written to {wti_out.name}")
    if fills_neg:
        print(f"      non-positive forward-fill ({len(fills_neg)}):")
        for d, old, new in fills_neg:
            print(f"        {d.date()}: {old:+.2f} -> {new:.2f}")
    if fills_wind:
        print(f"      Cushing-anomaly winsorization ({len(fills_wind)}):")
        for d, old, new in fills_wind:
            print(f"        {d.date()}: {old:.2f} -> {new:.2f}")

    # ---------- IT files (read existing) ----------
    psv_prices = load_dat(DATA / "gas_1826.dat")
    pun_prices = load_dat(DATA / "power_1826.dat")
    print(f"[PSV] {len(psv_prices)} obs (existing gas_1826.dat)")
    print(f"[PUN] {len(pun_prices)} obs (existing power_1826.dat)")

    # ---------- Describe ----------
    markets = [
        ("PSV gas", psv_prices),
        ("PUN power", pun_prices),
        ("PJM power", pjm_prices),
        ("WTI oil", wti_prices),
    ]
    results = [describe_market(name, p) for name, p in markets]

    header = (
        f"  {'Series':<14} "
        f"{'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} "
        f"{'Skew':>8} {'Kurt':>8}"
    )

    out_lines = []
    out_lines.append("=" * 90)
    out_lines.append("  DESCRIPTIVE STATISTICS - 4 ENERGY MARKETS")
    out_lines.append("=" * 90)

    for block, key, fmt in [
        ("RAW PRICES", "raw", "{:.3f}"),
        ("DETRENDED LOG-PRICES xi_t (LOESS frac=0.1)", "detrended", "{:.4f}"),
        ("LOG-RETURNS r_t = xi_t - xi_{t-1}", "returns", "{:.4f}"),
    ]:
        out_lines.append("")
        out_lines.append(f"-- {block} " + "-" * (87 - len(block)))
        out_lines.append(header)
        for r in results:
            out_lines.append(fmt_row(f"{r['name']} (N={r['n']})", r[key], fmt=fmt))

    out_lines.append("")
    out_lines.append("=" * 90)

    summary = "\n".join(out_lines)
    print()
    print(summary)

    summary_path = DATA / "preprocess_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n[summary written to {summary_path.name}]")


if __name__ == "__main__":
    main()
