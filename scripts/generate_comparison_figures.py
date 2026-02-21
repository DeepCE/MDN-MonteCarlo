#!/usr/bin/env python3
"""
Genera figure di confronto unconditional MDN vs parametriche.
- Genera 5000 traiettorie per Gas e Power su GPU in parallelo
- Salva i returns su MinIO per riuso futuro
- Crea figure di confronto
- Ricalcola e stampa momenti
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


def save_array(arr, filename):
    """Salva array numpy su MinIO."""
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    with fs.open_output_stream(f'ray-cluster/data/{filename}') as f:
        f.write(buf.read())
    print(f"  Saved: s3://ray-cluster/data/{filename}")


def save_figure(fig, filename):
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    with fs.open_output_stream(f'ray-cluster/figures/{filename}') as f:
        f.write(buf.read())
    print(f"  Saved figure: s3://ray-cluster/figures/{filename}")


def compute_moments(data):
    mean, std = np.mean(data), np.std(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data, fisher=False)  # Kurtosis non excess
    return {'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt}


@ray.remote(num_gpus=1)
def generate_trajectories(model_path, model_type, detrended, market_name, bin_range, n_traj=5000):
    """Genera traiettorie e calcola statistiche IN-PLACE (non restituisce array grandi)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{market_name}] Running on {device}")

    # Carica modello
    checkpoint = load_checkpoint(model_path, device)
    cfg = checkpoint['config']
    lookback = cfg['lookback']

    if model_type == 'base':
        model = AutoregressiveMDN(cfg['lookback'], cfg['hidden_dim'], cfg['n_layers'],
                                  cfg['n_components'], cfg.get('dropout', 0.1)).to(device)
    else:
        model = EnhancedMDN(cfg['lookback'], cfg['hidden_dim'], cfg['n_layers'],
                           cfg['n_components'], cfg['n_hidden_layers'], cfg['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_steps = len(detrended) - lookback
    initial = detrended[:lookback].copy()

    print(f"[{market_name}] Generating {n_traj} trajectories ({n_steps} steps each)...")

    # Calcola statistiche incrementalmente invece di accumulare tutti i returns
    # Per istogramma usiamo bins specifici per mercato (basati su ±5 sigma empirici)
    bins = np.linspace(-bin_range, bin_range, 71)
    histogram = np.zeros(len(bins) - 1)

    # Per momenti usiamo Welford's algorithm per stabilità numerica
    n_total = 0
    mean_acc = 0.0
    M2 = 0.0  # Per varianza
    M3 = 0.0  # Per skewness
    M4 = 0.0  # Per kurtosis

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

        # Calcola returns e aggiorna statistiche per ogni traiettoria
        for i in range(batch_n):
            returns = np.diff(trajectories[i])

            # Aggiorna istogramma
            hist_batch, _ = np.histogram(returns, bins=bins)
            histogram += hist_batch

            # Aggiorna momenti (Welford's online algorithm esteso)
            for r in returns:
                n_total += 1
                delta = r - mean_acc
                mean_acc += delta / n_total
                delta2 = r - mean_acc
                M2 += delta * delta2
                M3 += delta2**3 - 3 * delta2 * M2 / n_total
                M4 += delta2**4 - 4 * delta2 * M3 / n_total - 6 * delta2**2 * M2 / n_total**2

        if (batch_start + batch_n) % 1000 == 0:
            print(f"[{market_name}] Generated {batch_start + batch_n}/{n_traj} trajectories")

    # Calcola momenti finali
    variance = M2 / n_total
    std = np.sqrt(variance)
    skewness = (M3 / n_total) / (std ** 3) if std > 0 else 0
    kurtosis = (M4 / n_total) / (variance ** 2) if variance > 0 else 0

    print(f"[{market_name}] Total simulated returns: {n_total:,}")
    print(f"[{market_name}] Moments: Std={std:.4f}, Skew={skewness:.2f}, Kurt={kurtosis:.2f}")

    return {
        'market': market_name,
        'n_traj': n_traj,
        'n_returns': n_total,
        'n_components': cfg['n_components'],
        'histogram': histogram.tolist(),  # Lista piccola, non array enorme
        'bins': bins.tolist(),
        'moments': {
            'mean': mean_acc,
            'std': std,
            'skew': skewness,
            'kurt': kurtosis
        }
    }


def create_comparison_figure_from_histogram(emp_returns, sim_histogram, bins, market_name, bin_range):
    """Crea figura di confronto usando istogramma pre-calcolato."""

    # Fit distribuzioni parametriche sui dati EMPIRICI
    mu_emp, std_emp = np.mean(emp_returns), np.std(emp_returns)
    df_t, loc_t, scale_t = stats.t.fit(emp_returns)

    # Range per plot (specifico per mercato)
    x_range = np.linspace(-bin_range * 1.1, bin_range * 1.1, 500)

    # PDFs parametriche
    gauss_pdf = stats.norm.pdf(x_range, mu_emp, std_emp)
    t_pdf = stats.t.pdf(x_range, df_t, loc_t, scale_t)

    # Istogramma empirico
    hist_emp, _ = np.histogram(emp_returns, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Normalizza istogramma simulato a densità
    bin_width = bins[1] - bins[0]
    hist_sim = sim_histogram / (sim_histogram.sum() * bin_width)

    # === FIGURA ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel sinistro: scala lineare
    ax = axes[0]
    bar_width = bin_width * 0.9
    ax.bar(bin_centers, hist_emp, width=bar_width, alpha=0.4, color='black', label='Empirical')
    ax.plot(bin_centers, hist_sim, 'b-', lw=2.5, label='MDN Simulated')
    ax.plot(x_range, gauss_pdf, 'r--', lw=2, label='Gaussian')
    ax.plot(x_range, t_pdf, 'g--', lw=2, label=f't-Student (df={df_t:.1f})')
    ax.set_xlabel('Log-Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Unconditional Distribution Comparison', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-bin_range, bin_range)

    # Panel destro: scala logaritmica
    ax = axes[1]
    mask_emp = hist_emp > 0
    ax.scatter(bin_centers[mask_emp], hist_emp[mask_emp], c='black', s=40,
               marker='o', label='Empirical', zorder=5)
    mask_sim = hist_sim > 0
    ax.semilogy(bin_centers[mask_sim], hist_sim[mask_sim], 'b-', lw=2.5,
                label='MDN Simulated', zorder=4)
    ax.semilogy(x_range, gauss_pdf, 'r--', lw=2, label='Gaussian')
    ax.semilogy(x_range, t_pdf, 'g--', lw=2, label=f't-Student (df={df_t:.1f})')
    ax.set_xlabel('Log-Return', fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_title('Tail Behavior (Log Scale)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-bin_range, bin_range)
    ax.set_ylim(1e-4, 100)  # Abbassato limite inferiore per vedere meglio le code

    plt.suptitle(f'{market_name}: MDN vs Parametric Distributions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, df_t


if __name__ == "__main__":
    ray.init()
    print("=" * 70)
    print("  GENERAZIONE FIGURE CONFRONTO UNCONDITIONAL")
    print("  5000 traiettorie per commodity su GPU")
    print("=" * 70)

    # Carica dati
    print("\nCaricamento dati...")
    gas_prices = load_prices('gas_new_1826.dat')
    power_prices = load_prices('power_new_1826.dat')
    det_gas = loess_detrend(gas_prices)
    det_power = loess_detrend(power_prices)

    # Log-returns empirici (su TUTTA la serie, non solo dopo lookback)
    emp_returns_gas = np.diff(det_gas)
    emp_returns_power = np.diff(det_power)

    print(f"  Gas empirical returns: {len(emp_returns_gas)}")
    print(f"  Power empirical returns: {len(emp_returns_power)}")

    # Momenti empirici
    emp_mom_gas = compute_moments(emp_returns_gas)
    emp_mom_power = compute_moments(emp_returns_power)

    print(f"\n  Empirical moments:")
    print(f"    Gas:   Std={emp_mom_gas['std']:.4f}, Skew={emp_mom_gas['skew']:.2f}, Kurt={emp_mom_gas['kurt']:.2f}")
    print(f"    Power: Std={emp_mom_power['std']:.4f}, Skew={emp_mom_power['skew']:.2f}, Kurt={emp_mom_power['kurt']:.2f}")

    # Calcola bin ranges specifici per mercato (±5 sigma, arrotondato)
    bin_range_gas = round(5 * emp_mom_gas['std'], 2)  # ~0.35 per gas
    bin_range_power = round(5 * emp_mom_power['std'], 2)  # ~0.75 per power
    print(f"\n  Bin ranges: Gas=±{bin_range_gas}, Power=±{bin_range_power}")

    # Lancia job paralleli su 2 GPU
    print("\n  Launching 2 GPU workers...")
    gas_f = generate_trajectories.remote(
        'ray-cluster/models/mdn_gas_tuned.pt', 'base', det_gas, 'Natural Gas', bin_range_gas, n_traj=5000)
    power_f = generate_trajectories.remote(
        'ray-cluster/models/mdn_enhanced_power.pt', 'enhanced', det_power, 'Electric Power', bin_range_power, n_traj=5000)

    # Attendi risultati
    print("  Waiting for trajectory generation (this may take ~1 hour)...\n")
    gas_res = ray.get(gas_f)
    power_res = ray.get(power_f)

    # Estrai momenti e istogrammi (già calcolati nel worker)
    sim_mom_gas = gas_res['moments']
    sim_mom_power = power_res['moments']

    # Errori
    gas_std_err = (sim_mom_gas['std'] - emp_mom_gas['std']) / emp_mom_gas['std'] * 100
    gas_kurt_err = (sim_mom_gas['kurt'] - emp_mom_gas['kurt']) / emp_mom_gas['kurt'] * 100
    power_std_err = (sim_mom_power['std'] - emp_mom_power['std']) / emp_mom_power['std'] * 100
    power_kurt_err = (sim_mom_power['kurt'] - emp_mom_power['kurt']) / emp_mom_power['kurt'] * 100

    print("\n" + "=" * 70)
    print("  MOMENT VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n  Natural Gas ({gas_res['n_traj']} trajectories, {gas_res['n_returns']:,} returns):")
    print(f"    Empirical: Std={emp_mom_gas['std']:.4f}, Skew={emp_mom_gas['skew']:.2f}, Kurt={emp_mom_gas['kurt']:.2f}")
    print(f"    Simulated: Std={sim_mom_gas['std']:.4f}, Skew={sim_mom_gas['skew']:.2f}, Kurt={sim_mom_gas['kurt']:.2f}")
    print(f"    Errors:    Std={gas_std_err:+.1f}%, Kurt={gas_kurt_err:+.1f}%")

    print(f"\n  Electric Power ({power_res['n_traj']} trajectories, {power_res['n_returns']:,} returns):")
    print(f"    Empirical: Std={emp_mom_power['std']:.4f}, Skew={emp_mom_power['skew']:.2f}, Kurt={emp_mom_power['kurt']:.2f}")
    print(f"    Simulated: Std={sim_mom_power['std']:.4f}, Skew={sim_mom_power['skew']:.2f}, Kurt={sim_mom_power['kurt']:.2f}")
    print(f"    Errors:    Std={power_std_err:+.1f}%, Kurt={power_kurt_err:+.1f}%")

    # Crea e salva figure usando gli istogrammi pre-calcolati
    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)

    fig_gas, df_t_gas = create_comparison_figure_from_histogram(
        emp_returns_gas, np.array(gas_res['histogram']), np.array(gas_res['bins']), 'Natural Gas', bin_range_gas)
    save_figure(fig_gas, 'unconditional_comparison_natural_gas.png')
    plt.close(fig_gas)

    fig_power, df_t_power = create_comparison_figure_from_histogram(
        emp_returns_power, np.array(power_res['histogram']), np.array(power_res['bins']), 'Electric Power', bin_range_power)
    save_figure(fig_power, 'unconditional_comparison_electric_power.png')
    plt.close(fig_power)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Natural Gas:")
    print(f"    t-Student fitted df: {df_t_gas:.2f}")
    print(f"    Std Error: {gas_std_err:+.1f}%")
    print(f"    Kurt Error: {gas_kurt_err:+.1f}%")
    print(f"\n  Electric Power:")
    print(f"    t-Student fitted df: {df_t_power:.2f}")
    print(f"    Std Error: {power_std_err:+.1f}%")
    print(f"    Kurt Error: {power_kurt_err:+.1f}%")
    print(f"\n  Figures saved to s3://ray-cluster/figures/")
    print("=" * 70)

    ray.shutdown()
