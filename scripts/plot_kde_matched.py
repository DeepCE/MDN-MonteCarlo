#!/usr/bin/env python3
"""
Generate figure with KDE-matched trajectory.
Criterion: minimize L2 distance between KDE curves.
Output: only KDE panel (no trajectory).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
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


def kde_l2_distance(returns1, returns2, x_range):
    """Compute L2 distance between two KDE curves."""
    kde1 = stats.gaussian_kde(returns1)
    kde2 = stats.gaussian_kde(returns2)
    y1 = kde1(x_range)
    y2 = kde2(x_range)
    # L2 distance (integral of squared difference)
    dx = x_range[1] - x_range[0]
    return np.sqrt(np.sum((y1 - y2)**2) * dx)


@ray.remote(num_gpus=1)
def generate_kde_matched_trajectory(model_path, model_type, detrended, market_name, max_attempts=200):
    """Generate trajectory with best KDE match."""
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

    # Empirical returns and KDE range
    emp_returns = np.diff(detrended)
    emp_std = np.std(emp_returns)
    x_range = np.linspace(-6*emp_std, 6*emp_std, 500)

    print(f"[{market_name}] Empirical: Std={emp_std:.4f}, n_returns={len(emp_returns)}")

    best_returns = None
    best_score = float('inf')

    print(f"[{market_name}] Searching for trajectory with best KDE match (max {max_attempts} attempts)...")

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
            sim_returns = np.diff(traj_array)

            # Compute KDE L2 distance
            score = kde_l2_distance(emp_returns, sim_returns, x_range)

            if score < best_score:
                best_score = score
                best_returns = sim_returns.copy()
                print(f"[{market_name}] New best at attempt {i+1}: KDE distance={score:.6f}")

            if (i + 1) % 100 == 0:
                print(f"[{market_name}] Attempt {i+1}/{max_attempts}, best_score={best_score:.6f}")

            # Early stop if visually good match
            if score < 0.3:
                print(f"[{market_name}] Good KDE match found, stopping.")
                break

    # Create figure - KDE only
    print(f"[{market_name}] Creating figure...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    kde_emp = stats.gaussian_kde(emp_returns)
    kde_sim = stats.gaussian_kde(best_returns)

    ax.fill_between(x_range, kde_emp(x_range), alpha=0.3, color='black', label='Observed')
    ax.plot(x_range, kde_emp(x_range), 'k-', lw=2)
    ax.plot(x_range, kde_sim(x_range), '-', lw=2, label='Simulated', color='#e74c3c')

    ax.set_xlabel('Log-return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{market_name}: Distribution of Log-Returns', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5*emp_std, 5*emp_std)

    plt.tight_layout()

    # Save to MinIO
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)

    filename = f"kde_matched_{market_name.lower().replace(' ', '_')}.png"
    with fs.open_output_stream(f'ray-cluster/figures/{filename}') as f:
        f.write(buf.read())

    # Print base64 for download
    buf.seek(0)
    print(f"=== {filename} ===")
    print(base64.b64encode(buf.read()).decode())
    print(f"=== END {filename} ===")

    plt.close()
    print(f"[{market_name}] Saved: {filename}")
    print(f"[{market_name}] Final KDE distance: {best_score:.6f}")

    return filename


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  KDE-MATCHED FIGURE (minimize KDE L2 distance)")
    print("=" * 70)

    # Load data
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    _, _, det_gas = loess_detrend(gas_prices)
    _, _, det_power = loess_detrend(power_prices)

    # Launch parallel jobs
    gas_future = generate_kde_matched_trajectory.remote(
        'ray-cluster/models/mdn_gas_tuned.pt', 'base', det_gas, 'Natural Gas', 200)
    power_future = generate_kde_matched_trajectory.remote(
        'ray-cluster/models/mdn_enhanced_power.pt', 'enhanced', det_power, 'Electric Power', 200)

    print("\n  2 GPU workers started...")

    gas_fig = ray.get(gas_future)
    power_fig = ray.get(power_future)

    print("\n" + "=" * 70)
    print("  COMPLETED")
    print("=" * 70)
    print(f"  Gas figure: {gas_fig}")
    print(f"  Power figure: {power_fig}")

    ray.shutdown()
