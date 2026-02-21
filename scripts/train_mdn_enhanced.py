#!/usr/bin/env python3
"""
MDN Enhanced - Piu' componenti GMM e architettura piu' profonda.

Modifiche rispetto a train_mdn_final.py:
- n_components: 5 -> 10 (piu' componenti per catturare fat tails)
- Architettura: aggiunto secondo hidden layer con residual connection
- Dropout aumentato per regolarizzazione

Eseguire su cluster Ray con:
  ray job submit --address http://YOUR_RAY_CLUSTER:8265 --working-dir . --entrypoint-num-gpus 1 -- python train_mdn_enhanced.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

try:
    import pyarrow.fs as pafs
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False

# ============================================
# CONFIGURAZIONI MIGLIORATE
# ============================================

ENHANCED_CONFIGS = {
    'gas': {
        'lookback': 30,
        'hidden_dim': 128,
        'n_layers': 2,
        'n_components': 10,  # Aumentato da 5 a 10
        'n_hidden_layers': 2,  # Layers tra LSTM e MDN head
        'dropout': 0.15,
        'loess_frac': 0.1,
        'lr': 0.001,
        'batch_size': 256,
        'max_epochs': 400,
        'patience': 40,
    },
    'power': {
        'lookback': 30,
        'hidden_dim': 96,  # Leggermente aumentato
        'n_layers': 2,
        'n_components': 8,  # Aumentato da 5 a 8
        'n_hidden_layers': 2,
        'dropout': 0.15,
        'loess_frac': 0.1,
        'lr': 0.001,
        'batch_size': 256,
        'max_epochs': 400,
        'patience': 40,
    }
}

# ============================================
# DATA LOADING
# ============================================

def load_prices(filename):
    prices = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prices.append(float(line.replace(',', '.')))
                except:
                    continue
    return np.array(prices)


def loess_detrend(prices, frac=0.1):
    log_prices = np.log(prices)
    n = len(log_prices)
    window = max(5, int(n * frac) | 1)
    half_w = window // 2
    x = np.arange(-half_w, half_w + 1)
    sigma = half_w / 2
    weights = np.exp(-x**2 / (2 * sigma**2))
    weights /= weights.sum()
    padded = np.pad(log_prices, half_w, mode="edge")
    trend = np.array([np.sum(padded[i:i+window] * weights) for i in range(n)])
    detrended = log_prices - trend
    return log_prices, trend, detrended


# ============================================
# MODELLO ENHANCED
# ============================================

class EnhancedMDN(nn.Module):
    """
    MDN con architettura piu' profonda.

    Modifiche:
    - Multiple hidden layers tra LSTM e MDN head
    - Residual connections
    - Layer normalization
    - Piu' componenti GMM
    """

    def __init__(self, lookback=30, hidden_dim=128, n_layers=2,
                 n_components=10, n_hidden_layers=2, dropout=0.15):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_components = n_components

        # LSTM encoder
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)

        # Deep hidden layers con residual connections
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),  # GELU invece di ReLU per gradients piu' smooth
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Final projection before MDN heads
        self.pre_mdn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # MDN heads
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

        # Inizializzazione per sigma - bias negativo per sigma iniziali piccole
        nn.init.constant_(self.fc_sigma.bias, -1.0)

    def forward(self, x):
        # x: (batch, lookback, 1)
        h, _ = self.lstm(x)
        h = h[:, -1, :]  # (batch, hidden_dim)

        # Deep layers con residual connections
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            residual = h
            h = layer(h)
            h = norm(h + residual)  # Residual connection + LayerNorm

        # Pre-MDN projection
        h = self.pre_mdn(h)

        # MDN outputs
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        # Softplus con minimo per stabilita'
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4

        return pi, mu, sigma

    def sample(self, x, n_samples=1):
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)

        batch_size = x.size(0)
        device = x.device
        samples = []

        for _ in range(n_samples):
            components = torch.multinomial(pi, 1).squeeze(-1)
            batch_idx = torch.arange(batch_size, device=device)
            selected_mu = mu[batch_idx, components]
            selected_sigma = sigma[batch_idx, components]
            sample = torch.randn_like(selected_mu) * selected_sigma + selected_mu
            samples.append(sample)

        return torch.stack(samples, dim=-1).squeeze(-1) if n_samples == 1 else torch.stack(samples, dim=-1)


def mdn_loss(pi, mu, sigma, target):
    """Negative log-likelihood."""
    target = target.unsqueeze(-1)
    var = sigma ** 2
    log_prob = -0.5 * ((target - mu) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
    log_pi = torch.log(pi + 1e-10)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


# ============================================
# TRAINING
# ============================================

def train_model(model, train_loader, val_loader, config, device):
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['max_epochs']):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = mdn_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += mdn_loss(*model(xb), yb).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, lr={lr:.6f}" +
                  (" *" if patience_counter == 0 else ""))

        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return history, best_val_loss


# ============================================
# TRAJECTORY GENERATION & VALIDATION
# ============================================

def generate_trajectories(model, initial_history, n_steps, n_trajectories, device):
    model.eval()
    lookback = model.lookback
    histories = np.tile(initial_history, (n_trajectories, 1))
    trajectories = np.zeros((n_trajectories, n_steps))

    with torch.no_grad():
        for t in range(n_steps):
            x = torch.FloatTensor(histories).unsqueeze(-1).to(device)
            next_level = model.sample(x, n_samples=1).cpu().numpy()
            trajectories[:, t] = next_level
            histories = np.roll(histories, -1, axis=1)
            histories[:, -1] = next_level

    return trajectories


def compute_moments(data):
    mean = np.mean(data)
    std = np.std(data)
    skewness = np.mean(((data - mean) / std) ** 3)
    kurtosis = np.mean(((data - mean) / std) ** 4)
    return {'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurtosis}


def validate_moments(model, prices, detrended, trend, config, device, n_trajectories=1000):
    empirical_returns = np.diff(np.log(prices))
    emp_moments = compute_moments(empirical_returns)

    initial_history = detrended[-config['lookback']:]
    n_steps = len(prices) - 1

    detrended_traj = generate_trajectories(model, initial_history, n_steps, n_trajectories, device)

    last_trend = trend[-1]
    trend_growth = (trend[-1] - trend[0]) / len(trend)
    future_trend = last_trend + trend_growth * np.arange(1, n_steps + 1)
    price_traj = np.exp(detrended_traj + future_trend)

    all_gen_returns = []
    for i in range(n_trajectories):
        traj_returns = np.diff(np.log(price_traj[i]))
        all_gen_returns.extend(traj_returns)
    all_gen_returns = np.array(all_gen_returns)

    gen_moments = compute_moments(all_gen_returns)
    return emp_moments, gen_moments


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MDN ENHANCED - PIU' COMPONENTI E ARCHITETTURA PROFONDA")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\n  Caricamento dati...")
    gas_prices = load_prices('data/gas_1826.dat')
    power_prices = load_prices('data/power_1826.dat')
    print(f"    Gas: {len(gas_prices)} osservazioni")
    print(f"    Power: {len(power_prices)} osservazioni")

    results = {}

    for name, prices in [('gas', gas_prices), ('power', power_prices)]:
        print(f"\n{'='*70}")
        print(f"  TRAINING {name.upper()} (ENHANCED)")
        print("=" * 70)

        config = ENHANCED_CONFIGS[name]
        print(f"\n  Config:")
        print(f"    lookback={config['lookback']}, hidden={config['hidden_dim']}")
        print(f"    n_components={config['n_components']}, n_hidden_layers={config['n_hidden_layers']}")
        print(f"    dropout={config['dropout']}")

        # Detrending
        print("\n  [1/4] Detrending...")
        log_prices, trend, detrended = loess_detrend(prices, frac=config['loess_frac'])

        # Prepara dati
        print("\n  [2/4] Preparazione dataset...")
        lookback = config['lookback']
        X = np.array([detrended[i:i+lookback] for i in range(len(detrended)-lookback)])
        y = np.array([detrended[i+lookback] for i in range(len(detrended)-lookback)])

        split = int(0.8 * len(X))
        X_train = torch.FloatTensor(X[:split]).unsqueeze(-1)
        y_train = torch.FloatTensor(y[:split])
        X_val = torch.FloatTensor(X[split:]).unsqueeze(-1)
        y_val = torch.FloatTensor(y[split:])

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=config['batch_size'])

        print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

        # Crea modello
        print("\n  [3/4] Training...")
        model = EnhancedMDN(
            lookback=config['lookback'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_components=config['n_components'],
            n_hidden_layers=config['n_hidden_layers'],
            dropout=config['dropout']
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parametri: {n_params:,}")

        history, best_val_loss = train_model(model, train_loader, val_loader, config, device)
        print(f"\n    Best val_loss: {best_val_loss:.4f}")

        # Validazione momenti
        print("\n  [4/4] Validazione momenti (1000 traiettorie)...")
        emp_moments, gen_moments = validate_moments(
            model, prices, detrended, trend, config, device, n_trajectories=1000
        )

        print(f"\n  {'Momento':<12} {'Empirico':>12} {'Generato':>12} {'Errore':>10}")
        print(f"  {'-'*48}")

        std_err = abs(gen_moments['std'] - emp_moments['std']) / emp_moments['std'] * 100
        skew_diff = gen_moments['skewness'] - emp_moments['skewness']
        kurt_err = abs(gen_moments['kurtosis'] - emp_moments['kurtosis']) / emp_moments['kurtosis'] * 100

        print(f"  {'Std':<12} {emp_moments['std']:>12.6f} {gen_moments['std']:>12.6f} {std_err:>9.2f}%")
        print(f"  {'Skewness':<12} {emp_moments['skewness']:>12.4f} {gen_moments['skewness']:>12.4f} {skew_diff:>+9.2f}")
        print(f"  {'Kurtosis':<12} {emp_moments['kurtosis']:>12.4f} {gen_moments['kurtosis']:>12.4f} {kurt_err:>9.2f}%")

        results[name] = {
            'config': config,
            'val_loss': best_val_loss,
            'emp_moments': emp_moments,
            'gen_moments': gen_moments,
            'kurtosis_error': kurt_err,
            'std_error': std_err
        }

        # Salva modello
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'val_loss': best_val_loss,
            'emp_moments': emp_moments,
            'gen_moments': gen_moments,
            'trend_info': {
                'last_trend': trend[-1],
                'trend_growth': (trend[-1] - trend[0]) / len(trend),
            },
            'initial_history': detrended[-config['lookback']:],
        }

        local_path = f'mdn_enhanced_{name}.pt'
        torch.save(checkpoint, local_path)
        print(f"\n  Modello salvato: {local_path}")

        if HAS_MINIO:
            try:
                import io
                fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                minio_path = f"ray-cluster/models/mdn_enhanced_{name}.pt"
                with fs.open_output_stream(minio_path) as f:
                    f.write(buffer.read())
                print(f"  Salvato su MinIO: s3://{minio_path}")
            except Exception as e:
                print(f"  Warning: upload MinIO fallito: {e}")

    # Riepilogo finale
    print("\n" + "=" * 70)
    print("  RIEPILOGO ENHANCED vs PRECEDENTE")
    print("=" * 70)

    print(f"\n  RISULTATI ENHANCED:")
    print(f"  {'Commodity':<10} {'Val Loss':>10} {'Kurt Emp':>10} {'Kurt Gen':>10} {'Err Kurt':>10} {'Err Std':>10}")
    print(f"  {'-'*62}")
    for name, res in results.items():
        print(f"  {name.upper():<10} {res['val_loss']:>10.4f} "
              f"{res['emp_moments']['kurtosis']:>10.2f} "
              f"{res['gen_moments']['kurtosis']:>10.2f} "
              f"{res['kurtosis_error']:>9.2f}% "
              f"{res['std_error']:>9.2f}%")

    print(f"\n  CONFRONTO CON PRECEDENTE:")
    print(f"  {'Commodity':<10} {'Kurt Err Prec':>15} {'Kurt Err Now':>15} {'Miglioramento':>15}")
    print(f"  {'-'*57}")
    prev_errors = {'gas': 10.15, 'power': 3.44}
    for name, res in results.items():
        prev = prev_errors[name]
        now = res['kurtosis_error']
        improvement = prev - now
        symbol = "+" if improvement > 0 else ""
        print(f"  {name.upper():<10} {prev:>14.2f}% {now:>14.2f}% {symbol}{improvement:>14.2f}%")

    print("\n" + "=" * 70)
