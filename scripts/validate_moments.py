#!/usr/bin/env python3
"""Validazione momenti ensemble per entrambi i modelli a 8 componenti."""

import numpy as np
import torch
import torch.nn as nn
import io
import ray
import pyarrow.fs as pafs


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
    return lp - trend


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

    def sample(self, x):
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)
        comp = torch.multinomial(pi, 1).squeeze(-1)
        idx = torch.arange(x.size(0), device=x.device)
        return torch.randn_like(mu[idx, comp]) * sigma[idx, comp] + mu[idx, comp]


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

    def sample(self, x):
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)
        comp = torch.multinomial(pi, 1).squeeze(-1)
        idx = torch.arange(x.size(0), device=x.device)
        return torch.randn_like(mu[idx, comp]) * sigma[idx, comp] + mu[idx, comp]


def load_checkpoint(path, device):
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    with fs.open_input_file(path) as f:
        return torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)


def compute_moments(data):
    mean, std = np.mean(data), np.std(data)
    skew = np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    kurt = np.mean(((data - mean) / std) ** 4) if std > 0 else 0
    return {'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt}


@ray.remote(num_gpus=1)
def validate_model(model_path, model_type, detrended, market_name, n_traj=1000):
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
    initial = detrended[:lookback].copy()

    # Momenti empirici
    emp_returns = np.diff(detrended[lookback:])
    emp_mom = compute_moments(emp_returns)

    print(f"[{market_name}] Generating {n_traj} trajectories...")

    # Genera traiettorie
    all_returns = []
    batch_size = 100

    for batch_start in range(0, n_traj, batch_size):
        batch_n = min(batch_size, n_traj - batch_start)
        histories = np.tile(initial, (batch_n, 1))
        trajectories = np.zeros((batch_n, n_steps))

        with torch.no_grad():
            for t in range(n_steps):
                x = torch.FloatTensor(histories).unsqueeze(-1).to(device)
                next_val = model.sample(x).cpu().numpy()
                trajectories[:, t] = next_val
                histories = np.roll(histories, -1, axis=1)
                histories[:, -1] = next_val

        # Calcola returns per ogni traiettoria
        for i in range(batch_n):
            returns = np.diff(trajectories[i])
            all_returns.extend(returns)

    # Momenti simulati (ensemble)
    sim_returns = np.array(all_returns)
    sim_mom = compute_moments(sim_returns)

    # Errori
    std_err = (sim_mom['std'] - emp_mom['std']) / emp_mom['std'] * 100
    kurt_err = (sim_mom['kurt'] - emp_mom['kurt']) / emp_mom['kurt'] * 100

    print(f"\n[{market_name}] RESULTS ({n_traj} trajectories, {len(sim_returns)} returns):")
    print(f"  Empirical: Std={emp_mom['std']:.4f}, Skew={emp_mom['skew']:.2f}, Kurt={emp_mom['kurt']:.2f}")
    print(f"  Simulated: Std={sim_mom['std']:.4f}, Skew={sim_mom['skew']:.2f}, Kurt={sim_mom['kurt']:.2f}")
    print(f"  Errors:    Std={std_err:+.1f}%, Kurt={kurt_err:+.1f}%")

    return {
        'market': market_name,
        'n_components': cfg['n_components'],
        'n_traj': n_traj,
        'emp': emp_mom,
        'sim': sim_mom,
        'std_err': std_err,
        'kurt_err': kurt_err
    }


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  VALIDAZIONE MOMENTI ENSEMBLE")
    print("=" * 70)

    # Carica dati
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    det_gas = loess_detrend(gas_prices)
    det_power = loess_detrend(power_prices)

    # Valida entrambi i modelli (8 componenti)
    gas_f = validate_model.remote(
        'ray-cluster/models/mdn_fulldata_gas.pt', 'base', det_gas, 'Natural Gas', 1000)
    power_f = validate_model.remote(
        'ray-cluster/models/mdn_enhanced_power.pt', 'enhanced', det_power, 'Electric Power', 1000)

    print("\n  2 workers started...")

    gas_res = ray.get(gas_f)
    power_res = ray.get(power_f)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in [gas_res, power_res]:
        print(f"\n  {r['market']} ({r['n_components']} components, {r['n_traj']} trajectories):")
        print(f"    Empirical: Std={r['emp']['std']:.4f}, Skew={r['emp']['skew']:.2f}, Kurt={r['emp']['kurt']:.2f}")
        print(f"    Simulated: Std={r['sim']['std']:.4f}, Skew={r['sim']['skew']:.2f}, Kurt={r['sim']['kurt']:.2f}")
        print(f"    Errors:    Std={r['std_err']:+.1f}%, Kurt={r['kurt_err']:+.1f}%")
    print("=" * 70)
    ray.shutdown()
