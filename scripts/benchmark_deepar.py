#!/usr/bin/env python3
"""Benchmark DeepAR (gluonts / PyTorch backend).

Risposta a R2 punto 3 (benchmark probabilistico neurale con quantile forecasts).

Per ciascun mercato (PSV, PUN, PJM, WTI):
  1. Applica LOESS detrending -> serie xi_t.
  2. Split cronologico 80/20 (train / test).
  3. Fit DeepAR (LSTM + Student-t likelihood di default) su train.
  4. Genera N_TRAJ sample paths di lunghezza pari al test set.
  5. Calcola metriche probabilistiche via Evaluator:
       - mean wQuantileLoss (pinball loss media, risposta diretta a R2)
       - MSE, MAPE
  6. Salva: paths simulati + metriche + fit config.

Output:
  data/benchmarks/<market>_deepar_paths.npz
  data/benchmarks/deepar_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch

from gluonts.torch import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BENCH = DATA / "benchmarks"
BENCH.mkdir(parents=True, exist_ok=True)

N_TRAJ = 500           # sample paths per previsione
LOESS_FRAC = 0.10
MASTER_SEED = 42
CONTEXT_LEN = 30       # analogo al lookback m=30 del MDN
MAX_EPOCHS = 15        # training DeepAR (sufficiente per baseline)
BATCH_SIZE = 64
LR = 1e-3

MARKETS = [
    ("psv", "PSV gas",   "gas_1826.dat"),
    ("pun", "PUN power", "power_1826.dat"),
    ("pjm", "PJM power", "pjm_2451.dat"),
    ("wti", "WTI oil",   "wti_2501.dat"),
]


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


def make_dataset(series: np.ndarray, start: str = "2019-01-01", freq: str = "D"):
    """ListDataset from a 1D numpy array."""
    return ListDataset(
        [{"target": series.astype(np.float32), "start": pd.Period(start, freq=freq)}],
        freq=freq,
    )


def moments(x: np.ndarray) -> dict:
    m, s = float(np.mean(x)), float(np.std(x))
    if s == 0:
        return dict(std=0.0, skew=0.0, kurt=0.0)
    z = (x - m) / s
    return dict(std=s, skew=float(np.mean(z ** 3)), kurt=float(np.mean(z ** 4)))


def run_market(key: str, market: str, filename: str, summary_lines: list):
    print(f"\n[{market}] loading {filename} ...")
    prices = load_prices(DATA / filename)
    xi = loess_detrend(prices)

    # Split 80/20
    split = int(0.8 * len(xi))
    train = xi[:split]
    test = xi[split:]
    pred_len = len(test)
    print(f"[{market}] N={len(xi)}  train={len(train)}  test={len(test)}  "
          f"context={CONTEXT_LEN}")

    # Build DeepAR estimator. gluonts DeepAR defaults to Student-t likelihood,
    # which is a sensible parametric-neural baseline for heavy-tailed series.
    torch.manual_seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    estimator = DeepAREstimator(
        freq="D",
        prediction_length=pred_len,
        context_length=CONTEXT_LEN,
        num_layers=2,
        hidden_size=64,
        lr=LR,
        batch_size=BATCH_SIZE,
        num_batches_per_epoch=50,
        trainer_kwargs={
            "max_epochs": MAX_EPOCHS,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "accelerator": "cpu",
        },
    )

    # Fit
    print(f"[{market}] training DeepAR ({MAX_EPOCHS} epochs) ...")
    # gluonts vuole la serie intera; il test slicing lo fa make_evaluation_predictions
    full_ds = make_dataset(xi)
    train_ds = make_dataset(train)
    predictor = estimator.train(training_data=train_ds)

    # Evaluate on held-out test: make_evaluation_predictions uses last pred_len as holdout
    print(f"[{market}] generating {N_TRAJ} sample paths ...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_ds,
        predictor=predictor,
        num_samples=N_TRAJ,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    f = forecasts[0]
    samples = f.samples  # (N_TRAJ, pred_len)
    print(f"[{market}] samples shape: {samples.shape}")

    # Save
    out = BENCH / f"{key}_deepar_paths.npz"
    np.savez_compressed(
        out,
        paths=samples.astype(np.float64),
        train_xi=train.astype(np.float64),
        test_xi=test.astype(np.float64),
        context_length=np.array([CONTEXT_LEN]),
    )
    print(f"[{market}] saved {out.name} ({samples.nbytes/1e6:.1f} MB)")

    # Metrics (wQuantileLoss = pinball averaged over specified quantiles)
    evaluator = Evaluator(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    agg_metrics, _ = evaluator(tss, forecasts)
    wql = {q: agg_metrics.get(f"wQuantileLoss[{q}]", None)
           for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]}
    mean_wql = agg_metrics.get("mean_wQuantileLoss")
    mse = agg_metrics.get("MSE")
    crps = agg_metrics.get("mean_absolute_QuantileLoss")  # sum of QL
    msis = agg_metrics.get("MSIS")

    # Moment validation on sample paths vs empirical test
    r_emp = np.diff(test)
    r_sim_all = np.diff(samples, axis=1).ravel()
    r_emp_mom = moments(r_emp)
    r_sim_mom = moments(r_sim_all)

    summary_lines.append("")
    summary_lines.append(f"-- {market}  (N_train={len(train)}, N_test={len(test)}) "
                         + "-" * max(1, 50 - len(market)))
    summary_lines.append(f"  context_length={CONTEXT_LEN}  pred_length={pred_len}  "
                         f"num_samples={N_TRAJ}")
    summary_lines.append(f"  mean_wQuantileLoss = {mean_wql:.5f}")
    summary_lines.append(f"  MSE                = {mse:.5f}")
    for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        if wql.get(q) is not None:
            summary_lines.append(f"  wQL[{q:.2f}]        = {wql[q]:.5f}")
    summary_lines.append(
        f"  r_t test : std={r_emp_mom['std']:.4f}  skew={r_emp_mom['skew']:+.3f}  "
        f"kurt={r_emp_mom['kurt']:.2f}"
    )
    summary_lines.append(
        f"  r_t DeepAR: std={r_sim_mom['std']:.4f}  skew={r_sim_mom['skew']:+.3f}  "
        f"kurt={r_sim_mom['kurt']:.2f}"
    )

    return dict(mean_wql=mean_wql, mse=mse, wql=wql,
                r_emp=r_emp_mom, r_sim=r_sim_mom)


def main():
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("  DeepAR BENCHMARK (gluonts / PyTorch)")
    summary_lines.append(f"  context={CONTEXT_LEN}  epochs={MAX_EPOCHS}  "
                         f"num_samples={N_TRAJ}  LOESS_frac={LOESS_FRAC}")
    summary_lines.append("=" * 90)

    results = {}
    for key, market, filename in MARKETS:
        try:
            results[key] = run_market(key, market, filename, summary_lines)
        except Exception as e:
            summary_lines.append(f"\n  [ERROR] {market}: {e}")
            print(f"[{market}] ERROR: {e}")
            raise

    summary_lines.append("")
    summary_lines.append("=" * 90)
    summary_lines.append("Interpretation:")
    summary_lines.append("  mean_wQuantileLoss = pinball loss mediata su quantili [0.05..0.95].")
    summary_lines.append("  Piu' basso = migliore fit probabilistico su forecasting test set.")
    summary_lines.append("=" * 90)

    text = "\n".join(summary_lines)
    print("\n" + text)
    out = BENCH / "deepar_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[summary: {out}]")


if __name__ == "__main__":
    main()
