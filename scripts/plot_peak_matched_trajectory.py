#!/usr/bin/env python3
"""
Generate figure with peak-matched trajectory.
Simple criterion: match max and min of the trajectory.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import ray
import pyarrow.fs as pafs
import base64


def load_prices(fn):
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    with fs.open_input_file(f'ray-cluster/datasets/{fn}') as f:
        content = f.read().decode('utf-8')
    return np.array([float(l.strip().replace(',', '.')) for l in content.strip().split('\n') if l.strip()])


def loess_detrend(p, frac=0.1):
    lp = np.log(p)
    n = len(lp)
    w = max(5, int(n * frac) | 1)
    hw = w // 2
    wts = np.exp(-np.arange(-hw, hw + 1)**2 / (2 * (hw / 2)**2))
    wts /= wts.sum()
    pad = np.pad(lp, hw, mode="edge")
    trend = np.array([np.sum(pad[i:i + w] * wts) for i in range(n)])
    return lp, trend, lp - trend


class AutoregressiveMDN(nn.Module):
    def __init__(self, lookback=30, hidden_dim=128, n_layers=2, n_components=8, dropout=0.1):
        super().__init__()
        self.lookback, self.n_components = lookback, n_components
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_hidden = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.fc_hidden(self.lstm(x)[0][:, -1, :])
        return (torch.softmax(self.fc_pi(h), -1),
                self.fc_mu(h),
                torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4)


class EnhancedMDN(nn.Module):
    def __init__(self, lookback=30, hidden_dim=96, n_layers=2, n_components=8, n_hidden_layers=2, dropout=0.15):
        super().__init__()
        self.lookback, self.n_components = lookback, n_components
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_hidden_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden_layers)])
        self.pre_mdn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            h = norm(layer(h) + h)
        h = self.pre_mdn(h)
        return (torch.softmax(self.fc_pi(h), -1),
                self.fc_mu(h),
                torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4)


def load_checkpoint(path, device):
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    with fs.open_input_file(path) as f:
        return torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)


def sample_gmm(pi, mu, sigma):
    k = torch.multinomial(pi, 1).item()
    return torch.normal(mu[k], sigma[k]).item()


@ray.remote(num_gpus=1)
def generate_peak_matched_trajectory(model_path, model_type, detrended, market_name, max_attempts=500):
    """Generate trajectory matching max and min of empirical data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{market_name}] Loading model on {device}...")

    checkpoint = load_checkpoint(model_path, device)
    cfg = checkpoint['config']

    if model_type == 'base':
        model = AutoregressiveMDN(cfg['lookback'], cfg['hidden_dim'], cfg['n_layers'],
                                  cfg['n_components'], cfg.get('dropout', 0.1)).to(device)
    else:
        model = EnhancedMDN(cfg['lookback'], cfg['hidden_dim'], cfg['n_layers'],
                           cfg['n_components'], cfg['n_hidden_layers'], cfg['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    lookback = cfg['lookback']
    n_steps = len(detrended) - lookback

    # Empirical max and min
    emp_max = np.max(detrended)
    emp_min = np.min(detrended)

    print(f"[{market_name}] Empirical: Max={emp_max:.4f}, Min={emp_min:.4f}")

    best_traj = None
    best_score = float('inf')

    print(f"[{market_name}] Searching for trajectory matching max/min (max {max_attempts} attempts)...")

    with torch.no_grad():
        for i in range(max_attempts):
            # Generate one trajectory
            trajectory = detrended[:lookback].copy().tolist()
            for _ in range(n_steps):
                x = torch.FloatTensor(trajectory[-lookback:]).unsqueeze(0).unsqueeze(-1).to(device)
                pi, mu, sigma = model(x)
                next_val = sample_gmm(pi[0], mu[0], sigma[0])
                trajectory.append(next_val)

            traj_array = np.array(trajectory)
            traj_max = np.max(traj_array)
            traj_min = np.min(traj_array)

            # Score: relative error on max and min
            max_err = abs(traj_max - emp_max) / abs(emp_max)
            min_err = abs(traj_min - emp_min) / abs(emp_min)
            score = max_err + min_err

            if score < best_score:
                best_score = score
                best_traj = traj_array.copy()
                print(f"[{market_name}] New best at attempt {i+1}: score={score:.4f} "
                      f"(max={traj_max:.4f}, min={traj_min:.4f})")

            if (i + 1) % 100 == 0:
                print(f"[{market_name}] Attempt {i+1}/{max_attempts}, best_score={best_score:.4f}")

            # Early stop if very good match (both within 10%)
            if max_err < 0.10 and min_err < 0.10:
                print(f"[{market_name}] Good match found (both <10% error), stopping.")
                break

    # Create figure - trajectories only
    print(f"[{market_name}] Creating figure...")

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))

    days = np.arange(len(detrended))

    ax1.plot(days, detrended, 'k-', lw=1.5, label='Observed', alpha=0.9)
    ax1.plot(days, best_traj, '-', lw=1.2, label='Simulated (peak-matched)', alpha=0.8, color='#9b59b6')

    ax1.set_xlabel('Days', fontsize=11)
    ax1.set_ylabel(r'Detrended log-price $\xi_t$', fontsize=11)
    ax1.set_title(f'{market_name}: Observed vs Peak-Matched Simulated Trajectory', fontsize=13, fontweight='bold')
    # Legend removed - colors described in caption
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(detrended)-1)

    plt.tight_layout()

    # Save to MinIO
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)

    filename = f"trajectory_peak_matched_{market_name.lower().replace(' ', '_')}.png"
    with fs.open_output_stream(f'ray-cluster/figures/{filename}') as f:
        f.write(buf.read())

    # Print base64 for download
    buf.seek(0)
    print(f"=== {filename} ===")
    print(base64.b64encode(buf.read()).decode())
    print(f"=== END {filename} ===")

    plt.close()
    print(f"[{market_name}] Saved: {filename}")
    print(f"[{market_name}] Final score: {best_score:.4f}")

    return filename


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  PEAK-MATCHED TRAJECTORY FIGURE (max/min matching)")
    print("=" * 70)

    # Load data
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    _, _, det_gas = loess_detrend(gas_prices)
    _, _, det_power = loess_detrend(power_prices)

    # Launch parallel jobs
    gas_future = generate_peak_matched_trajectory.remote(
        'ray-cluster/models/mdn_gas_tuned.pt', 'base', det_gas, 'Natural Gas', 500)
    power_future = generate_peak_matched_trajectory.remote(
        'ray-cluster/models/mdn_enhanced_power.pt', 'enhanced', det_power, 'Electric Power', 500)

    print("\n  2 GPU workers started...")

    gas_fig = ray.get(gas_future)
    power_fig = ray.get(power_future)

    print("\n" + "=" * 70)
    print("  COMPLETED")
    print("=" * 70)
    print(f"  Gas figure: {gas_fig}")
    print(f"  Power figure: {power_fig}")

    ray.shutdown()
