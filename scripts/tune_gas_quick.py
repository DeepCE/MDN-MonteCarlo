#!/usr/bin/env python3
"""
Tuning rapido Gas: prova diverse configurazioni per migliorare 8 componenti.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


def mdn_loss(pi, mu, sigma, target):
    target = target.unsqueeze(-1)
    log_probs = -0.5 * ((target - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_probs = log_probs + torch.log(pi + 1e-10)
    return -torch.logsumexp(log_probs, dim=-1).mean()


def compute_moments(data):
    mean, std = np.mean(data), np.std(data)
    skew = np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    kurt = np.mean(((data - mean) / std) ** 4) if std > 0 else 0
    return {'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt}


@ray.remote(num_gpus=1)
def train_and_validate(config, detrended, config_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config_name}] Starting on {device}...")

    lookback = config['lookback']
    n_components = config['n_components']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    dropout = config['dropout']
    model_type = config.get('model_type', 'base')

    # Prepara dati
    X, y = [], []
    for i in range(lookback, len(detrended) - 1):
        X.append(detrended[i - lookback:i])
        y.append(detrended[i])
    X = np.array(X)
    y = np.array(y)

    # Split train/val (90/10)
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.FloatTensor(X_train).unsqueeze(-1), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val).unsqueeze(-1), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # Crea modello
    if model_type == 'enhanced':
        model = EnhancedMDN(lookback, hidden_dim, n_layers, n_components,
                           config.get('n_hidden_layers', 2), dropout).to(device)
    else:
        model = AutoregressiveMDN(lookback, hidden_dim, n_layers, n_components, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config['max_epochs']):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pi, mu, sigma = model(batch_x)
            loss = mdn_loss(pi, mu, sigma, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pi, mu, sigma = model(batch_x)
                val_loss += mdn_loss(pi, mu, sigma, batch_y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"[{config_name}] Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Validate con traiettorie
    print(f"[{config_name}] Generating validation trajectories...")
    n_traj = 500
    n_steps = len(detrended) - lookback
    initial = detrended[:lookback].copy()

    emp_returns = np.diff(detrended[lookback:])
    emp_mom = compute_moments(emp_returns)

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

        for i in range(batch_n):
            returns = np.diff(trajectories[i])
            all_returns.extend(returns)

    sim_returns = np.array(all_returns)
    sim_mom = compute_moments(sim_returns)

    std_err = (sim_mom['std'] - emp_mom['std']) / emp_mom['std'] * 100
    kurt_err = (sim_mom['kurt'] - emp_mom['kurt']) / emp_mom['kurt'] * 100

    # Score: vogliamo minimizzare gli errori assoluti
    score = abs(std_err) + abs(kurt_err)

    print(f"[{config_name}] DONE: Std={std_err:+.1f}%, Kurt={kurt_err:+.1f}%, Score={score:.1f}")

    return {
        'config_name': config_name,
        'config': config,
        'val_loss': best_val_loss,
        'emp': emp_mom,
        'sim': sim_mom,
        'std_err': std_err,
        'kurt_err': kurt_err,
        'score': score,
        'model_state': best_state
    }


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  TUNING RAPIDO GAS - Configurazioni")
    print("=" * 70)

    # Carica dati
    gas_prices = load_prices('gas_new_1826.dat')
    det_gas = loess_detrend(gas_prices)

    # Configurazioni da testare
    configs = [
        # Base con più dropout
        {
            'name': '8comp_d015',
            'lookback': 30, 'hidden_dim': 128, 'n_layers': 2, 'n_components': 8,
            'dropout': 0.15, 'lr': 0.001, 'batch_size': 256, 'max_epochs': 200, 'patience': 25,
            'model_type': 'base'
        },
        # Base con ancora più dropout
        {
            'name': '8comp_d020',
            'lookback': 30, 'hidden_dim': 128, 'n_layers': 2, 'n_components': 8,
            'dropout': 0.20, 'lr': 0.001, 'batch_size': 256, 'max_epochs': 200, 'patience': 25,
            'model_type': 'base'
        },
        # Enhanced (come Power)
        {
            'name': '8comp_enhanced',
            'lookback': 30, 'hidden_dim': 96, 'n_layers': 2, 'n_components': 8,
            'n_hidden_layers': 2, 'dropout': 0.15, 'lr': 0.001, 'batch_size': 256,
            'max_epochs': 200, 'patience': 25, 'model_type': 'enhanced'
        },
        # 6 componenti (trade-off)
        {
            'name': '6comp_base',
            'lookback': 30, 'hidden_dim': 128, 'n_layers': 2, 'n_components': 6,
            'dropout': 0.15, 'lr': 0.001, 'batch_size': 256, 'max_epochs': 200, 'patience': 25,
            'model_type': 'base'
        },
        # 7 componenti
        {
            'name': '7comp_base',
            'lookback': 30, 'hidden_dim': 128, 'n_layers': 2, 'n_components': 7,
            'dropout': 0.15, 'lr': 0.001, 'batch_size': 256, 'max_epochs': 200, 'patience': 25,
            'model_type': 'base'
        },
        # Enhanced con 6 componenti
        {
            'name': '6comp_enhanced',
            'lookback': 30, 'hidden_dim': 96, 'n_layers': 2, 'n_components': 6,
            'n_hidden_layers': 2, 'dropout': 0.15, 'lr': 0.001, 'batch_size': 256,
            'max_epochs': 200, 'patience': 25, 'model_type': 'enhanced'
        },
    ]

    # Lancia tutti i job in parallelo (3 GPU, 6 config -> 2 round)
    futures = []
    for cfg in configs:
        name = cfg.pop('name')
        futures.append(train_and_validate.remote(cfg, det_gas, name))

    print(f"\n  {len(configs)} configurazioni lanciate su 3 GPU...")

    results = ray.get(futures)

    # Ordina per score
    results.sort(key=lambda x: x['score'])

    print("\n" + "=" * 70)
    print("  RISULTATI (ordinati per score)")
    print("=" * 70)

    for r in results:
        print(f"\n  {r['config_name']} (n_comp={r['config']['n_components']}, type={r['config'].get('model_type', 'base')}):")
        print(f"    Empirical: Std={r['emp']['std']:.4f}, Kurt={r['emp']['kurt']:.2f}")
        print(f"    Simulated: Std={r['sim']['std']:.4f}, Kurt={r['sim']['kurt']:.2f}")
        print(f"    Errors:    Std={r['std_err']:+.1f}%, Kurt={r['kurt_err']:+.1f}%")
        print(f"    Score:     {r['score']:.1f}")

    # Salva il migliore
    best = results[0]
    print(f"\n  MIGLIORE: {best['config_name']} con score {best['score']:.1f}")

    # Salva modello
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    checkpoint = {
        'config': best['config'],
        'model_state_dict': best['model_state']
    }
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    buf.seek(0)
    with fs.open_output_stream('ray-cluster/models/mdn_gas_tuned.pt') as f:
        f.write(buf.read())
    print(f"  Salvato: s3://ray-cluster/models/mdn_gas_tuned.pt")

    print("=" * 70)
    ray.shutdown()
