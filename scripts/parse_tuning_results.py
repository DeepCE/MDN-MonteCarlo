#!/usr/bin/env python3
"""Parse Ray Tune results e estrai best config per ciascun mercato.

Il tuning (mdn_tuning_minio_v2.py) salva i risultati via Ray Tune. Dopo il
completamento, questo script li legge (via Jobs API logs o results_df dump) e
produce best_configs.json con la best configuration per ciascun mercato
(minimizzando best_val_loss).

Uso:
  # Se eseguito sul cluster (accesso diretto ai risultati):
  python code/parse_tuning_results.py --from-cluster <ray-job-id>

  # Se eseguito in locale e i risultati sono in un CSV/json dumpato:
  python code/parse_tuning_results.py --from-csv tuning_results.csv

  # Fallback: legge i log del job Ray e fa parsing dei Trial result blocks
  python code/parse_tuning_results.py --from-job-logs <ray-job-id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_job_logs(job_id: str, address: str) -> list[dict]:
    """Esegue `ray job logs <job_id>` e ne estrae i trial completati.

    Ritorna lista di dict con: commodity, lookback, hidden_dim, n_layers,
    n_components, lr, dropout, best_val_loss.
    """
    env = {"PYTHONIOENCODING": "utf-8", "RAY_ADDRESS": address}
    cmd = ["py", "-3.11", "-m", "ray.scripts.scripts", "job", "logs", job_id]
    res = subprocess.run(cmd, capture_output=True, text=True, env={**__import__('os').environ, **env})
    text = res.stdout + res.stderr

    trials = []
    # Trial config blocks
    cfg_blocks = re.finditer(
        r"Trial (train_mdn_[^\s]+) config\s+│.*?├.*?│\s*batch_size\s*│\s*(\d+)\s*│\s*"
        r"│\s*commodity\s*│\s*(\w+)\s*│\s*"
        r"│\s*dropout\s*│\s*([\d.]+)\s*│\s*"
        r"│\s*hidden_dim\s*│\s*(\d+)\s*│\s*"
        r"│\s*lookback\s*│\s*(\d+)\s*│\s*"
        r"│\s*lr\s*│\s*([\d.e-]+)\s*│\s*"
        r"│\s*max_epochs\s*│\s*\d+\s*│\s*"
        r"│\s*n_components\s*│\s*(\d+)\s*│\s*"
        r"│\s*n_layers\s*│\s*(\d+)\s*│",
        text, re.DOTALL,
    )
    configs = {}
    for m in cfg_blocks:
        tid = m.group(1)
        configs[tid] = dict(
            batch_size=int(m.group(2)),
            commodity=m.group(3),
            dropout=float(m.group(4)),
            hidden_dim=int(m.group(5)),
            lookback=int(m.group(6)),
            lr=float(m.group(7)),
            n_components=int(m.group(8)),
            n_layers=int(m.group(9)),
        )

    # Trial result blocks con best_val_loss
    res_blocks = re.finditer(
        r"Trial (train_mdn_[^\s]+) result\s+│.*?"
        r"│\s*best_val_loss\s*│\s*([+-]?[\d.]+)\s*│",
        text, re.DOTALL,
    )
    for m in res_blocks:
        tid = m.group(1)
        if tid in configs:
            trials.append({**configs[tid], "trial_id": tid,
                           "best_val_loss": float(m.group(2))})

    return trials


def pick_best(trials: list[dict]) -> dict:
    best = {}
    for t in trials:
        c = t["commodity"]
        if c not in best or t["best_val_loss"] < best[c]["best_val_loss"]:
            best[c] = t
    # Drop trial_id, keep only config fields
    out = {}
    for c, t in best.items():
        out[c] = {
            "lookback": t["lookback"],
            "hidden_dim": t["hidden_dim"],
            "n_layers": t["n_layers"],
            "n_components": t["n_components"],
            "lr": t["lr"],
            "dropout": t["dropout"],
            "batch_size": t["batch_size"],
            "best_val_loss": t["best_val_loss"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-job-logs", help="Ray job id")
    ap.add_argument("--address", default=os.environ.get("RAY_ADDRESS", "http://localhost:8265"))
    ap.add_argument("--output", default=str(ROOT / "best_configs.json"))
    args = ap.parse_args()

    if args.from_job_logs:
        print(f"Parsing logs of job {args.from_job_logs} ...")
        trials = parse_job_logs(args.from_job_logs, args.address)
        print(f"Found {len(trials)} completed trials")
    else:
        print("usage: --from-job-logs <RAY_JOB_ID>")
        return

    best = pick_best(trials)
    print(f"\nBest configs (by best_val_loss):")
    for c, cfg in best.items():
        print(f"  {c}: {cfg}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
