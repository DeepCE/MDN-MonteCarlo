#!/usr/bin/env python3
"""
Visualizzazione evoluzione distribuzioni GMM nel tempo.
Mostra come la distribuzione condizionale cambia in diversi momenti della serie.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
import io
import ray
import pyarrow.fs as pafs


def load_prices(fn):
    """Carica prezzi da MinIO."""
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


def save_figure(fig, filename):
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    with fs.open_output_stream(f'ray-cluster/figures/{filename}') as f:
        f.write(buf.read())


def gmm_pdf(x, pi, mu, sigma):
    """Calcola PDF della GMM."""
    pdf = np.zeros_like(x)
    for k in range(len(pi)):
        pdf += pi[k] * stats.norm.pdf(x, mu[k], sigma[k])
    return pdf


@ray.remote(num_gpus=1)
def plot_gmm_evolution(model_path, model_type, detrended, market_name, bin_range):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{market_name}] Loading model on {device}... (x-range: ±{bin_range})")

    # Carica modello
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
    n_components = cfg['n_components']

    # Calcola log-returns empirici
    log_returns = np.diff(detrended)

    # Fitta t-Student INCONDIZIONATA su tutti i returns (questa e' la distribuzione statica)
    t_df, t_loc, t_scale = stats.t.fit(log_returns)
    print(f"[{market_name}] Unconditional t-Student: df={t_df:.2f}, loc={t_loc:.4f}, scale={t_scale:.4f}")

    # Identifica momenti interessanti
    # 1. Periodo calmo (bassa volatilita' rolling)
    # 2. Pre-shock (prima di un evento estremo)
    # 3. Durante shock (alta volatilita')
    # 4. Post-shock (ritorno alla normalita')

    window = 30
    rolling_std = np.array([np.std(log_returns[max(0, i-window):i+1])
                           for i in range(len(log_returns))])

    # Trova indici
    calm_idx = lookback + np.argmin(rolling_std[lookback:-100])  # Periodo piu' calmo
    volatile_idx = lookback + np.argmax(rolling_std[lookback:-100])  # Periodo piu' volatile

    # Pre e post shock (30 giorni prima/dopo il picco di volatilita')
    pre_shock_idx = max(lookback, volatile_idx - 30)
    post_shock_idx = min(len(detrended) - 1, volatile_idx + 30)

    timepoints = {
        'Calm Period': calm_idx,
        'Pre-Shock': pre_shock_idx,
        'High Volatility': volatile_idx,
        'Post-Shock': post_shock_idx
    }

    print(f"[{market_name}] Selected timepoints:")
    for name, idx in timepoints.items():
        print(f"  {name}: day {idx}, rolling_std={rolling_std[min(idx, len(rolling_std)-1)]:.4f}")

    # Estrai distribuzioni GMM per ogni timepoint
    # NOTA: La GMM predice xi_{t+1}. Per ottenere r_{t+1} = xi_{t+1} - xi_t,
    # trasliamo le medie di -xi_t (il valore corrente)
    gmm_params = {}
    current_values = {}
    with torch.no_grad():
        for name, idx in timepoints.items():
            # Prepara input (lookback finestra)
            x = torch.FloatTensor(detrended[idx-lookback:idx]).unsqueeze(0).unsqueeze(-1).to(device)
            pi, mu, sigma = model(x)
            # Valore corrente xi_t (ultimo della finestra)
            xi_t = detrended[idx-1]
            current_values[name] = xi_t
            # Trasla le medie per ottenere distribuzione dei RETURNS
            mu_returns = mu[0].cpu().numpy() - xi_t
            gmm_params[name] = {
                'pi': pi[0].cpu().numpy(),
                'mu': mu_returns,  # Traslate per i returns
                'sigma': sigma[0].cpu().numpy()
            }

    # Crea figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, idx) in zip(axes, timepoints.items()):
        params = gmm_params[name]
        pi, mu, sigma = params['pi'], params['mu'], params['sigma']

        # Calcola range locale basato sulla volatilità del periodo (±5 sigma locale)
        local_std = rolling_std[min(idx, len(rolling_std)-1)]
        local_range = max(5 * local_std, 0.15)  # Minimo 0.15 per evitare range troppo stretti
        x_range_local = np.linspace(-local_range * 1.1, local_range * 1.1, 500)

        # Plot GMM totale (distribuzione dei RETURNS)
        gmm_total = gmm_pdf(x_range_local, pi, mu, sigma)
        ax.fill_between(x_range_local, gmm_total, alpha=0.3, color='blue', label='MDN (conditional)')
        ax.plot(x_range_local, gmm_total, 'b-', lw=2)

        # Plot t-Student INCONDIZIONATA (stessa per tutti i periodi)
        t_pdf = stats.t.pdf(x_range_local, t_df, t_loc, t_scale)
        ax.plot(x_range_local, t_pdf, 'r--', lw=2, label=f't-Student (df={t_df:.1f})')

        # Annotazioni
        ax.axvline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('Log-Return')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} (Day {idx}, σ={local_std:.3f})', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-local_range, local_range)

    plt.suptitle(f'{market_name}: MDN Conditional vs Static t-Student Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fn = f"gmm_evolution_{market_name.lower().replace(' ', '_')}.png"
    save_figure(fig, fn)
    plt.close()
    print(f"[{market_name}] Saved: {fn}")

    # === SECONDA FIGURA: Confronto GMM vs Parametriche ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Usa periodo volatile per confronto (piu' interessante)
    volatile_idx = timepoints['High Volatility']
    params = gmm_params['High Volatility']
    pi, mu, sigma = params['pi'], params['mu'], params['sigma']

    # Range basato sulla volatilità del periodo High Volatility
    volatile_std = rolling_std[min(volatile_idx, len(rolling_std)-1)]
    volatile_range = max(5 * volatile_std, 0.2)
    x_range_volatile = np.linspace(-volatile_range * 1.1, volatile_range * 1.1, 500)

    # Plot 1: Confronto distribuzioni
    ax = axes2[0]
    gmm_total = gmm_pdf(x_range_volatile, pi, mu, sigma)
    ax.fill_between(x_range_volatile, gmm_total, alpha=0.3, color='blue')
    ax.plot(x_range_volatile, gmm_total, 'b-', lw=2, label='MDN (conditional)')

    # Gaussiana globale
    gauss_pdf_global = stats.norm.pdf(x_range_volatile, np.mean(log_returns), np.std(log_returns))
    ax.plot(x_range_volatile, gauss_pdf_global, 'g--', lw=2, label='Gaussian')

    # t-Student fitted (incondizionata - usa parametri gia' calcolati)
    t_pdf = stats.t.pdf(x_range_volatile, t_df, t_loc, t_scale)
    ax.plot(x_range_volatile, t_pdf, 'r--', lw=2, label=f't-Student (df={t_df:.1f})')

    ax.set_xlabel('Log-Return')
    ax.set_ylabel('Density')
    ax.set_title(f'MDN vs Parametric (High Volatility, σ={volatile_std:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-volatile_range, volatile_range)

    # Plot 2: Log-scale per vedere le code
    ax = axes2[1]
    ax.semilogy(x_range_volatile, gmm_total, 'b-', lw=2, label='MDN (conditional)')
    ax.semilogy(x_range_volatile, gauss_pdf_global, 'g--', lw=2, label='Gaussian')
    ax.semilogy(x_range_volatile, t_pdf, 'r--', lw=2, label=f't-Student (df={t_df:.1f})')

    # Istogramma empirico con bins adattati al range volatile
    hist_bins = np.linspace(-volatile_range, volatile_range, 51)
    counts, _ = np.histogram(log_returns, bins=hist_bins, density=True)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    ax.scatter(bin_centers, counts + 1e-6, c='black', s=20, alpha=0.5, label='Empirical')

    ax.set_xlabel('Log-Return')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Tail Behavior Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-volatile_range, volatile_range)
    ax.set_ylim(1e-4, 50)

    plt.suptitle(f'{market_name}: MDN Captures What Parametric Models Cannot', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fn2 = f"gmm_vs_parametric_{market_name.lower().replace(' ', '_')}.png"
    save_figure(fig2, fn2)
    plt.close()
    print(f"[{market_name}] Saved: {fn2}")

    return {
        'market': market_name,
        'timepoints': timepoints,
        'n_components': n_components,
        'figures': [fn, fn2]
    }


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  VISUALIZZAZIONE EVOLUZIONE GMM")
    print("=" * 70)

    # Carica dati da MinIO
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    _, _, det_gas = loess_detrend(gas_prices)
    _, _, det_power = loess_detrend(power_prices)

    # Calcola bin ranges specifici per mercato (±5 sigma)
    gas_returns = np.diff(det_gas)
    power_returns = np.diff(det_power)
    bin_range_gas = round(5 * np.std(gas_returns), 2)
    bin_range_power = round(5 * np.std(power_returns), 2)
    print(f"\n  Bin ranges: Gas=±{bin_range_gas}, Power=±{bin_range_power}")

    # Lancia job paralleli
    gas_f = plot_gmm_evolution.remote(
        'ray-cluster/models/mdn_gas_tuned.pt', 'base', det_gas, 'Natural Gas', bin_range_gas)
    power_f = plot_gmm_evolution.remote(
        'ray-cluster/models/mdn_enhanced_power.pt', 'enhanced', det_power, 'Electric Power', bin_range_power)

    print("\n  2 workers started...")

    gas_res = ray.get(gas_f)
    power_res = ray.get(power_f)

    print("\n" + "=" * 70)
    print("  COMPLETED")
    print("=" * 70)
    for r in [gas_res, power_res]:
        print(f"\n  {r['market']} ({r['n_components']} components):")
        for name, idx in r['timepoints'].items():
            print(f"    {name}: day {idx}")
        print(f"    Figures: {', '.join(r['figures'])}")

    print("=" * 70)
    ray.shutdown()
