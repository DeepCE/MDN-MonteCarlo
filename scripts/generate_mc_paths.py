#!/usr/bin/env python3
"""Generazione 5000 MC paths MDN per ciascun mercato, autoregressive.

Ogni path:
  - lunghezza pari al dataset empirico (T obs)
  - seed = ultimi m osservazioni della serie detrended (come da paper originale)
  - campionamento autoregressive dalla GMM condizionale ad ogni step

Output:
  data/mdn_paths/<market>_mdn_paths.npz  contenente paths (N_TRAJ x T) + metadata
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdn_models import load_checkpoint_model  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = DATA / "mdn_paths"
OUT.mkdir(parents=True, exist_ok=True)

COMMODITY_FILES = {
    "psv": "gas_1826.dat",
    "pun": "power_1826.dat",
    "pjm": "pjm_2451.dat",
    "wti": "wti_2501.dat",
}

# Mappatura variante -> {market: checkpoint filename in models/}.
# 'retrained' : ckpt ri-addestrati (aprile 2026, config paper ma nuovo seed).
# 'paper_k8'  : ckpt K=8 dichiarato nel paper (mdn_fulldata_gas.pt, mdn_enhanced_power.pt).
# 'paper_tab3': ckpt i cui gen_moments memorizzati combaciano con Tab.3 del paper.
#               Per PSV e' mdn_final_gas.pt (K=5), per PUN coincide con paper_k8.
# 'v2'        : ckpt prodotti dal walk-forward tuning v2 (train_final_v2.py).
MODEL_MAP = {
    "retrained": {
        "psv": "mdn_final_psv.pt",
        "pun": "mdn_final_pun.pt",
        "pjm": "mdn_final_pjm.pt",
        "wti": "mdn_final_wti.pt",
    },
    "paper_k8": {
        "psv": "mdn_fulldata_gas.pt",
        "pun": "mdn_enhanced_power.pt",
    },
    "paper_tab3": {
        "psv": "mdn_final_gas.pt",          # K=5: gen_moments coincide con Tab.3
        "pun": "mdn_enhanced_power.pt",      # come paper_k8
    },
    "v2": {
        "psv": "mdn_v2_psv.pt",
        "pun": "mdn_v2_pun.pt",
        "pjm": "mdn_v2_pjm.pt",
        "wti": "mdn_v2_wti.pt",
    },
}

# Suffisso applicato al file di output npz per tenere separate le varianti.
VARIANT_SUFFIX = {
    "retrained": "",
    "paper_k8": "_paper_k8",
    "paper_tab3": "_paper_tab3",
    "v2": "_v2",
}

N_TRAJ_DEFAULT = 5000   # coerente con tabelle v1 del paper.
                        # Bande [P5,P95] ~0.3pp di errore sul percentile.
                        # Su GPU ~1-2 min per mercato; su CPU ~20 min.
MASTER_SEED = 42

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_BUCKET = "ray-cluster"
MINIO_PATHS_PREFIX = "data/mdn_paths"


def upload_minio(local_path: Path, remote_key: str) -> None:
    """Upload a file to MinIO if pyarrow is available; silent-skip on failure."""
    try:
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem(endpoint_override=MINIO_ENDPOINT)
        remote_full = f"{MINIO_BUCKET}/{remote_key}"
        with open(local_path, "rb") as src, fs.open_output_stream(remote_full) as dst:
            dst.write(src.read())
        print(f"    [minio] uploaded -> s3://{remote_full}  ({local_path.stat().st_size/1e6:.2f} MB)")
    except Exception as e:
        print(f"    [minio] WARN upload failed ({type(e).__name__}): {e}")


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


# AutoregressiveMDN / EnhancedMDN imported via mdn_models.load_checkpoint_model


@torch.no_grad()
def simulate(model, initial_history: np.ndarray, n_steps: int, n_traj: int,
             device: torch.device, seed: int) -> np.ndarray:
    m = model.lookback
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    histories = torch.tensor(np.tile(initial_history, (n_traj, 1)), dtype=torch.float32,
                             device=device)  # (n_traj, m)
    paths = np.empty((n_traj, n_steps), dtype=np.float32)

    # batch size: full batch su GPU, 2000 max su CPU per evitare swap
    batch = n_traj if device.type == "cuda" else min(n_traj, 2000)
    for start in range(0, n_traj, batch):
        end = min(start + batch, n_traj)
        hb = histories[start:end].clone()
        for t in range(n_steps):
            x = hb.unsqueeze(-1)  # (b, m, 1)
            pi, mu, sigma = model(x)
            # sample component
            comp = torch.multinomial(pi, 1).squeeze(-1)
            idx = torch.arange(pi.size(0), device=device)
            mu_sel = mu[idx, comp]
            sigma_sel = sigma[idx, comp]
            eps = torch.randn_like(mu_sel)
            next_val = mu_sel + sigma_sel * eps
            paths[start:end, t] = next_val.cpu().numpy()
            hb = torch.cat([hb[:, 1:], next_val.unsqueeze(-1)], dim=1)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="psv,pun,pjm,wti")
    ap.add_argument("--n_traj", type=int, default=N_TRAJ_DEFAULT)
    ap.add_argument("--variant", choices=sorted(MODEL_MAP.keys()), default="retrained",
                    help="quale set di checkpoint usare: 'retrained' (default) o 'paper'")
    ap.add_argument("--out-suffix", default=None,
                    help="override del suffisso npz; default = VARIANT_SUFFIX[variant]")
    args = ap.parse_args()

    mapping = MODEL_MAP[args.variant]
    suffix = args.out_suffix if args.out_suffix is not None else VARIANT_SUFFIX[args.variant]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  variant: {args.variant}  |  out suffix: '{suffix}'")

    for market in args.markets.split(","):
        ckpt_name = mapping.get(market)
        if ckpt_name is None:
            print(f"[{market}] SKIP (variant '{args.variant}' has no checkpoint for this market)")
            continue
        ckpt_path = MODELS / ckpt_name
        if not ckpt_path.exists():
            print(f"[{market}] SKIP (no checkpoint at {ckpt_path})")
            continue
        print(f"\n[{market}] loading {ckpt_path.name} ...")
        model, ckpt = load_checkpoint_model(ckpt_path, device)

        prices = load_dat(DATA / COMMODITY_FILES[market])
        xi = loess_detrend(prices)
        m = ckpt.get("lookback") or ckpt["config"]["lookback"]
        initial = ckpt.get("initial_history")
        if initial is None:
            initial = xi[-m:]
        else:
            initial = np.asarray(initial, dtype=np.float32).reshape(-1)[-m:]
        n_steps = len(xi)  # path lungo tutto il dataset per confronto diretto

        print(f"[{market}] simulating {args.n_traj} paths x {n_steps} steps "
              f"(m={m}, K={ckpt['config']['n_components']}) ...")
        t0 = time.time()
        seed = MASTER_SEED + abs(hash(market)) % 10_000
        paths = simulate(model, initial, n_steps, args.n_traj, device, seed)
        dt = time.time() - t0
        print(f"[{market}] done in {dt:.1f}s  shape={paths.shape}  mem={paths.nbytes/1e6:.1f} MB")

        out = OUT / f"{market}_mdn_paths{suffix}.npz"
        np.savez_compressed(
            out,
            paths=paths,
            initial_history=initial,
            empirical_xi=xi.astype(np.float32),
            config=np.array([ckpt["config"]], dtype=object),
            market=market,
            seed=seed,
            gen_time_s=dt,
            variant=args.variant,
            ckpt_name=ckpt_name,
        )
        print(f"[{market}] saved {out.name}")
        upload_minio(out, f"{MINIO_PATHS_PREFIX}/{out.name}")


if __name__ == "__main__":
    main()
