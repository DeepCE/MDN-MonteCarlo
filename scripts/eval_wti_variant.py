#!/usr/bin/env python3
"""Eval-only companion to run_wti_variants.py.

Loads a saved checkpoint and runs VaR + PIT diagnostics (no MC generation).
Writes summary to data/wti_variant_{mode}_summary.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from mdn_models import load_checkpoint_model  # noqa: E402
from run_wti_variants import (  # noqa: E402
    CFG, DATA, loess_detrend, load_dat, forward_all, run_var, run_pit,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("moment", "cushing", "tstudent"), required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt_path = ROOT / "models" / f"mdn_v2_wti_{args.mode}.pt"
    model, ckpt = load_checkpoint_model(ckpt_path, device)
    cfg = ckpt["config"]
    lam = ckpt.get("lambda_moment")
    best = ckpt.get("phase1_best_val")
    print(f"loaded {ckpt_path.name}  config={cfg}  best={best:+.4f}  lambda={lam}",
          flush=True)

    xi = loess_detrend(load_dat(DATA / "wti_2501.dat"))
    PI, MU, SIGMA = forward_all(model, xi, device)

    print("\n--- VaR backtest ---", flush=True)
    var_res = run_var(PI, MU, SIGMA, xi)
    print("\n--- PIT diagnostics ---", flush=True)
    pit_res = run_pit(PI, MU, SIGMA, xi)

    lines = []
    lines.append("=" * 104)
    lines.append(f"  WTI variant -- {args.mode} (eval-only)")
    lines.append(f"  Config: {cfg}")
    lines.append(f"  Phase 1 best val = {best:+.4f}")
    if lam is not None:
        lines.append(f"  lambda (moment) = {lam}   target kurt = 31.96")
    lines.append("=" * 104)
    lines.append("\n[VaR backtest]")
    lines.append(f"  {'level':<10} {'tail':<6} {'N':>5} {'exc':>5} {'rate':>7} "
                 f"{'exp':>5} {'p_uc':>7} {'p_ind':>7} {'p_cc':>7}")
    for key, r in var_res.items():
        lvl, tail = key.split("_")
        lines.append(
            f"  {lvl:<10} {tail:<6} {r['n']:>5d} {r['exceed']:>5d} "
            f"{r['rate']:>7.4f} {r['expected']:>5.2f} "
            f"{r['p_uc']:>7.4f} {r['p_ind']:>7.4f} {r['p_cc']:>7.4f}"
        )
    lines.append("\n[PIT diagnostics]")
    ks_s, ks_p = pit_res["ks"]; ad_s, ad_p = pit_res["ad"]
    lb_s, lb_p = pit_res["lb"]; bk_s, bk_p, bk_mu, bk_rho, bk_sig = pit_res["bk"]
    lines.append(f"  N={pit_res['n']}  u_mean={pit_res['u_mean']:.4f}  u_std={pit_res['u_std']:.4f}")
    lines.append(f"  KS  stat={ks_s:.4f}  p={ks_p:.4f}")
    lines.append(f"  AD  stat={ad_s:.3f}  p={ad_p}")
    lines.append(f"  LB  stat={lb_s:.2f}   p={lb_p:.4f}")
    lines.append(f"  Berk LR={bk_s:.2f}  p={bk_p:.4f}   "
                 f"mu={bk_mu:+.3f}  rho={bk_rho:+.3f}  sigma={bk_sig:.3f}")
    lines.append("\n[Reference: baseline Gaussian MDN]")
    lines.append("  VaR: 1/4 pass CC; left rates 0.44% (alpha=0.01), 2.74% (alpha=0.05)")
    lines.append("  PIT: u_mean=0.58, sigma_Berk=0.87")
    lines.append("=" * 104)

    text = "\n".join(lines); print("\n" + text)
    out = DATA / f"wti_variant_{args.mode}_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
