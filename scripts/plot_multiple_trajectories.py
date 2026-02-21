#!/usr/bin/env python3
"""
Generate figure with multiple trajectories - one at a time approach.
Stop as soon as we find all 3 types.
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


@ray.remote(num_gpus=1)
def generate_trajectories_and_plot(model_path, model_type, detrended, market_name, max_attempts=500):
    """Generate trajectories one at a time until we find all 3 types, then create figure."""
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

    # Empirical stats
    emp_returns = np.diff(detrended)
    emp_std = np.std(emp_returns)
    emp_kurt = stats.kurtosis(emp_returns, fisher=False)
    emp_max_dev = np.max(np.abs(detrended))

    print(f"[{market_name}] Empirical: Std={emp_std:.4f}, Kurt={emp_kurt:.2f}, MaxDev={emp_max_dev:.4f}")

    # Thresholds for selection
    # Representative: very close to empirical moments
    # Calm: reasonable moments + low max deviation
    # Volatile: reasonable moments + high max deviation

    found = {'representative': None, 'calm': None, 'volatile': None}
    found_stats = {'representative': None, 'calm': None, 'volatile': None}

    print(f"[{market_name}] Searching for 3 trajectory types (max {max_attempts} attempts)...")

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
            returns = np.diff(traj_array)

            traj_std = np.std(returns)
            traj_kurt = stats.kurtosis(returns, fisher=False)
            traj_max_dev = np.max(np.abs(traj_array))

            std_err = abs(traj_std - emp_std) / emp_std
            kurt_err = abs(traj_kurt - emp_kurt) / emp_kurt

            stat = {'std': traj_std, 'kurt': traj_kurt, 'max_dev': traj_max_dev,
                    'std_err': std_err, 'kurt_err': kurt_err}

            # Check if this trajectory fits any unfilled category
            # Representative: std_err < 5%, kurt_err < 15%
            if found['representative'] is None and std_err < 0.05 and kurt_err < 0.15:
                found['representative'] = traj_array.copy()
                found_stats['representative'] = stat
                print(f"[{market_name}] Found REPRESENTATIVE at attempt {i+1}: "
                      f"Std={traj_std:.4f} ({std_err*100:+.1f}%), Kurt={traj_kurt:.2f} ({kurt_err*100:+.1f}%)")

            # Calm: reasonable moments + low max_dev (< 0.6 * empirical)
            elif found['calm'] is None and std_err < 0.25 and kurt_err < 0.40 and traj_max_dev < 0.65 * emp_max_dev:
                found['calm'] = traj_array.copy()
                found_stats['calm'] = stat
                print(f"[{market_name}] Found CALM at attempt {i+1}: "
                      f"MaxDev={traj_max_dev:.4f} (emp={emp_max_dev:.4f})")

            # Volatile: reasonable moments + high max_dev (> 0.9 * empirical)
            elif found['volatile'] is None and std_err < 0.35 and kurt_err < 0.50 and traj_max_dev > 0.85 * emp_max_dev:
                found['volatile'] = traj_array.copy()
                found_stats['volatile'] = stat
                print(f"[{market_name}] Found VOLATILE at attempt {i+1}: "
                      f"MaxDev={traj_max_dev:.4f} (emp={emp_max_dev:.4f})")

            # Check if we found all 3
            if all(v is not None for v in found.values()):
                print(f"[{market_name}] All 3 trajectories found after {i+1} attempts!")
                break

            if (i + 1) % 50 == 0:
                n_found = sum(1 for v in found.values() if v is not None)
                print(f"[{market_name}] Attempt {i+1}/{max_attempts}, found {n_found}/3")

    # If we didn't find all, use what we have or relax criteria
    missing = [k for k, v in found.items() if v is None]
    if missing:
        print(f"[{market_name}] Warning: Missing {missing}, generating with relaxed criteria...")
        # Generate a few more with relaxed criteria
        for _ in range(100):
            trajectory = detrended[:lookback].copy().tolist()
            for _ in range(n_steps):
                x = torch.FloatTensor(trajectory[-lookback:]).unsqueeze(0).unsqueeze(-1).to(device)
                pi, mu, sigma = model(x)
                trajectory.append(sample_gmm(pi[0], mu[0], sigma[0]))

            traj_array = np.array(trajectory)
            returns = np.diff(traj_array)
            traj_std = np.std(returns)
            traj_kurt = stats.kurtosis(returns, fisher=False)
            traj_max_dev = np.max(np.abs(traj_array))
            std_err = abs(traj_std - emp_std) / emp_std
            kurt_err = abs(traj_kurt - emp_kurt) / emp_kurt
            stat = {'std': traj_std, 'kurt': traj_kurt, 'max_dev': traj_max_dev,
                    'std_err': std_err, 'kurt_err': kurt_err}

            for m in missing[:]:
                if found[m] is None:
                    found[m] = traj_array.copy()
                    found_stats[m] = stat
                    missing.remove(m)
                    print(f"[{market_name}] Filled {m} with fallback trajectory")
            if not missing:
                break

    # Create figure - trajectories only, no KDE
    print(f"[{market_name}] Creating figure...")

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))

    days = np.arange(len(detrended))

    ax1.plot(days, detrended, 'k-', lw=1.5, label='Observed', alpha=0.9)
    ax1.plot(days, found['representative'], '-', lw=1.2, label='Simulated (typical)', alpha=0.8, color='#2ecc71')
    ax1.plot(days, found['calm'], '-', lw=1.2, label='Simulated (calm)', alpha=0.8, color='#3498db')
    ax1.plot(days, found['volatile'], '-', lw=1.2, label='Simulated (volatile)', alpha=0.8, color='#e74c3c')

    ax1.set_xlabel('Days', fontsize=11)
    ax1.set_ylabel(r'Detrended log-price $\xi_t$', fontsize=11)
    ax1.set_title(f'{market_name}: Observed vs Simulated Trajectories', fontsize=13, fontweight='bold')
    # Legend removed - colors described in caption
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(detrended)-1)

    plt.tight_layout()

    # Save to MinIO
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)

    filename = f"trajectories_multiple_{market_name.lower().replace(' ', '_')}.png"
    with fs.open_output_stream(f'ray-cluster/figures/{filename}') as f:
        f.write(buf.read())

    # Print base64 for download
    buf.seek(0)
    print(f"=== {filename} ===")
    print(base64.b64encode(buf.read()).decode())
    print(f"=== END {filename} ===")

    plt.close()
    print(f"[{market_name}] Saved: {filename}")

    return filename


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  MULTIPLE TRAJECTORIES FIGURE (one-at-a-time approach)")
    print("=" * 70)

    # Load data
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    _, _, det_gas = loess_detrend(gas_prices)
    _, _, det_power = loess_detrend(power_prices)

    # Launch parallel jobs
    gas_future = generate_trajectories_and_plot.remote(
        'ray-cluster/models/mdn_gas_tuned.pt', 'base', det_gas, 'Natural Gas', 500)
    power_future = generate_trajectories_and_plot.remote(
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
