#!/usr/bin/env python3
"""Estrai il best_epoch dal log del tuning Ray per ogni mercato.

Parsing line-based dei Trial config + result blocks (DOTALL regex era
troppo lento su log da 19k righe).

Output: aggiorna best_configs.json con max_epochs per ciascun mercato
(= total epochs del trial vincente, che include patience dopo best_val_loss).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_PATH = ROOT / "best_configs.json"

PATIENCE = 25  # from mdn_tuning_minio_v2.py


# Ray box uses Unicode │ (U+2502), not ASCII |
B = r"[|\u2502]"  # either pipe char

FIELD_PATTERNS = {
    "batch_size": re.compile(rf"{B}\s*batch_size\s+(\d+)\s*{B}"),
    "commodity": re.compile(rf"{B}\s*commodity\s+(\w+)\s*{B}"),
    "dropout": re.compile(rf"{B}\s*dropout\s+([\d.]+)\s*{B}"),
    "hidden_dim": re.compile(rf"{B}\s*hidden_dim\s+(\d+)\s*{B}"),
    "lookback": re.compile(rf"{B}\s*lookback\s+(\d+)\s*{B}"),
    "lr": re.compile(rf"{B}\s*lr\s+([\d.eE+-]+)\s*{B}"),
    "n_components": re.compile(rf"{B}\s*n_components\s+(\d+)\s*{B}"),
    "n_layers": re.compile(rf"{B}\s*n_layers\s+(\d+)\s*{B}"),
}

RESULT_PATTERNS = {
    "training_iteration": re.compile(rf"{B}\s*training_iteration\s+(\d+)\s*{B}"),
    "best_val_loss": re.compile(rf"{B}\s*best_val_loss\s+([+-]?[\d.]+)\s*{B}"),
    "epoch": re.compile(rf"{B}\s*epoch\s+(\d+)\s*{B}"),
    "val_loss": re.compile(rf"{B}\s*val_loss\s+([+-]?[\d.]+)\s*{B}"),
}

TRIAL_CFG_START = re.compile(r"Trial (train_mdn_\w+) config")
TRIAL_RES_START = re.compile(
    r"Trial (train_mdn_\w+) completed after (\d+) iterations"
)


def parse_block(lines: list[str], i: int, patterns: dict, max_lines: int = 25) -> dict:
    """Dal punto i, leggi fino a max_lines o fino a linea che chiude il box,
    estrai i campi via patterns."""
    out = {}
    end = min(i + max_lines, len(lines))
    for j in range(i, end):
        ln = lines[j]
        for k, pat in patterns.items():
            if k in out:
                continue
            m = pat.search(ln)
            if m:
                out[k] = m.group(1)
        if all(k in out for k in patterns):
            break
    return out


def parse_trials(log_text: str) -> list[dict]:
    lines = log_text.split("\n")
    configs = {}
    results = {}
    for i, ln in enumerate(lines):
        mc = TRIAL_CFG_START.search(ln)
        if mc:
            tid = mc.group(1)
            block = parse_block(lines, i, FIELD_PATTERNS, max_lines=20)
            if "commodity" in block:
                configs[tid] = {
                    "batch_size": int(block.get("batch_size", 0)),
                    "commodity": block["commodity"],
                    "dropout": float(block["dropout"]),
                    "hidden_dim": int(block["hidden_dim"]),
                    "lookback": int(block["lookback"]),
                    "lr": float(block["lr"]),
                    "n_components": int(block["n_components"]),
                    "n_layers": int(block["n_layers"]),
                }
            continue
        mr = TRIAL_RES_START.search(ln)
        if mr:
            tid = mr.group(1)
            iters = int(mr.group(2))
            block = parse_block(lines, i, RESULT_PATTERNS, max_lines=20)
            if "best_val_loss" in block:
                results[tid] = {
                    "iterations": iters,
                    "training_iteration": int(block.get("training_iteration", iters)),
                    "best_val_loss": float(block["best_val_loss"]),
                    "final_epoch": int(block.get("epoch", iters - 1)),
                    "val_loss": float(block.get("val_loss", 0.0)),
                }

    trials = []
    for tid, cfg in configs.items():
        if tid in results:
            trials.append({**cfg, **results[tid], "trial_id": tid})
    return trials


def pick_best(trials: list[dict]) -> dict[str, dict]:
    best = {}
    for t in trials:
        c = t["commodity"]
        if c not in best or t["best_val_loss"] < best[c]["best_val_loss"]:
            best[c] = t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", nargs="?", default="/tmp/tuning_logs.txt")
    ap.add_argument("--configs", default=str(CONFIGS_PATH))
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()

    log_path = Path(args.log_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"Loaded {len(text):,} bytes from {log_path}")

    trials = parse_trials(text)
    print(f"Parsed {len(trials)} completed trials")

    best = pick_best(trials)
    if not best:
        print("ERROR: no best trials found")
        return

    configs_path = Path(args.configs)
    cfg = json.loads(configs_path.read_text()) if configs_path.exists() else {}

    print(f"\n{'market':<6} {'iters':>6} {'best_epoch':>11} {'best_val_loss':>15}  trial")
    for c, t in best.items():
        iters = t["iterations"]
        best_epoch = max(1, iters - args.patience)
        print(f"{c:<6} {iters:>6d} {best_epoch:>11d} {t['best_val_loss']:>+15.4f}  {t['trial_id']}")
        cfg.setdefault(c, {})
        cfg[c]["best_val_loss"] = t["best_val_loss"]
        cfg[c]["total_tuning_epochs"] = iters
        cfg[c]["max_epochs"] = iters  # full-data budget = same as tuning total

    configs_path.write_text(json.dumps(cfg, indent=2))
    print(f"\nWritten {configs_path}")


if __name__ == "__main__":
    main()
