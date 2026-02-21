#!/usr/bin/env python3
"""
Genera traiettorie rappresentative con momenti simili a quelli empirici.

Per ogni commodity:
1. Genera 1000 traiettorie Monte Carlo
2. Calcola i momenti di ciascuna (std, skewness, kurtosis)
3. Seleziona la traiettoria con momenti piu' vicini a quelli storici
4. Plotta confronto visivo: time series + tabella momenti

Eseguire su cluster Ray:
  ray job submit --address http://YOUR_RAY_CLUSTER:8265 --working-dir . -- python plot_representative_trajectory.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import classi modello
from generative_mdn import AutoregressiveMDN, load_prices, loess_detrend, generate_trajectories

# Import EnhancedMDN per Power
import sys
import importlib.util

# Carica EnhancedMDN da train_mdn_enhanced.py
spec = importlib.util.spec_from_file_location("train_mdn_enhanced", "train_mdn_enhanced.py")
enhanced_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_module)
EnhancedMDN = enhanced_module.EnhancedMDN

# ============================================
# CONFIGURAZIONE
# ============================================

N_TRAJECTORIES = 5000  # Traiettorie da generare (piu' sono, meglio matcha i picchi)
SEED = 42

# Modelli finali (da research log)
# Ordine di ricerca: finale locale -> base locale -> MinIO
MODEL_PATHS = {
    'gas': ['mdn_final_gas.pt', 'mdn_model_gas.pt'],
    'power': ['mdn_enhanced_power.pt', 'mdn_model_power.pt'],
}

MINIO_PATHS = {
    'gas': 'ray-cluster/models/mdn_final_gas.pt',
    'power': 'ray-cluster/models/mdn_enhanced_power.pt',
}

# Nota: per power preferiamo EnhancedMDN se disponibile, altrimenti fallback a base
MODEL_CLASSES = {
    'gas': AutoregressiveMDN,
    'power': EnhancedMDN,
}

# ============================================
# FUNZIONI
# ============================================

def compute_moments(returns):
    """Calcola i 4 momenti di una serie di returns."""
    mean = np.mean(returns)
    std = np.std(returns)
    if std < 1e-10:
        return {'mean': mean, 'std': std, 'skewness': 0, 'kurtosis': 3}

    z = (returns - mean) / std
    skewness = np.mean(z ** 3)
    kurtosis = np.mean(z ** 4)

    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def compute_trajectory_moments(price_trajectory):
    """Calcola momenti dei log-returns di una traiettoria."""
    log_returns = np.diff(np.log(price_trajectory))
    return compute_moments(log_returns)


def select_representative_trajectory(trajectories, empirical_moments, prices,
                                      detrended_traj, detrended_obs):
    """
    Seleziona la traiettoria con momenti E forma piu' vicini a quelli empirici.

    Criterio: minimizza errore relativo pesato su:
    - std e kurtosis (momenti)
    - min e max della serie detrended (picchi estremi)
    """
    n_traj = trajectories.shape[0]

    # Statistiche empiriche sui detrended
    obs_min = detrended_obs.min()
    obs_max = detrended_obs.max()
    obs_range = obs_max - obs_min

    # Calcola momenti e statistiche per ogni traiettoria
    traj_moments = []
    traj_stats = []
    for i in range(n_traj):
        moments = compute_trajectory_moments(trajectories[i])
        traj_moments.append(moments)

        # Statistiche sui detrended
        traj_min = detrended_traj[i].min()
        traj_max = detrended_traj[i].max()
        traj_stats.append({'min': traj_min, 'max': traj_max})

    # Calcola score (errore relativo pesato)
    scores = []
    for i, (m, s) in enumerate(zip(traj_moments, traj_stats)):
        # Errore relativo su std (peso 1)
        std_err = abs(m['std'] - empirical_moments['std']) / empirical_moments['std']

        # Errore relativo su kurtosis (peso 1)
        kurt_err = abs(m['kurtosis'] - empirical_moments['kurtosis']) / empirical_moments['kurtosis']

        # Errore assoluto su skewness (peso 0.3)
        skew_err = abs(m['skewness'] - empirical_moments['skewness'])

        # Errore sui picchi estremi (peso 1.5 - importante per visual match)
        min_err = abs(s['min'] - obs_min) / obs_range
        max_err = abs(s['max'] - obs_max) / obs_range

        score = std_err + kurt_err + 0.3 * skew_err + 1.5 * (min_err + max_err)
        scores.append(score)

    # Seleziona traiettoria con score minimo
    best_idx = np.argmin(scores)

    # Aggiungi info sui picchi ai momenti
    best_moments = traj_moments[best_idx].copy()
    best_moments['min'] = traj_stats[best_idx]['min']
    best_moments['max'] = traj_stats[best_idx]['max']

    return best_idx, best_moments, scores[best_idx]


def load_model(commodity):
    """
    Carica modello cercando in ordine:
    1. File locali (finale, poi base)
    2. MinIO
    """
    checkpoint = None
    source = None

    # Prova file locali
    for path in MODEL_PATHS[commodity]:
        try:
            checkpoint = torch.load(path, weights_only=False, map_location='cpu')
            source = f"local:{path}"
            print(f"    Loaded from {path}")
            break
        except FileNotFoundError:
            continue

    # Fallback a MinIO
    if checkpoint is None:
        try:
            import pyarrow.fs as pafs
            import io
            fs = pafs.S3FileSystem(
                endpoint_override="http://YOUR_MINIO_HOST:9000",
                access_key=os.environ.get("MINIO_ACCESS_KEY"),
                secret_key=os.environ.get("MINIO_SECRET_KEY")
            )
            minio_path = MINIO_PATHS[commodity]
            with fs.open_input_file(minio_path) as f:
                buffer = io.BytesIO(f.read())
            checkpoint = torch.load(buffer, weights_only=False, map_location='cpu')
            source = f"minio:{minio_path}"
            print(f"    Loaded from MinIO: {minio_path}")
        except Exception as e:
            raise RuntimeError(f"Cannot load model for {commodity}: {e}")

    config = checkpoint['config']

    # Determina classe modello in base alla config
    # Se ha n_hidden_layers, usa EnhancedMDN
    if config.get('n_hidden_layers'):
        model = EnhancedMDN(
            lookback=config['lookback'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_components=config['n_components'],
            n_hidden_layers=config.get('n_hidden_layers', 2),
            dropout=config.get('dropout', 0.15)
        )
    else:
        model = AutoregressiveMDN(
            lookback=config['lookback'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_components=config['n_components']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, source


def generate_price_trajectories(model, detrended, trend, config, n_trajectories):
    """Genera traiettorie di prezzi."""
    lookback = config['lookback']
    n_steps = len(detrended) - lookback

    # Usa gli stessi primi 'lookback' valori come storia iniziale
    initial_history = detrended[:lookback]

    # Genera traiettorie detrended
    detrended_traj = generate_trajectories(model, initial_history, n_steps, n_trajectories)

    # Converti a prezzi usando il trend osservato
    price_traj = np.zeros_like(detrended_traj)
    for i in range(n_trajectories):
        log_synth = detrended_traj[i] + trend[lookback:]
        price_traj[i] = np.exp(log_synth)

    return price_traj, detrended_traj


def plot_comparison(commodity, prices, representative_traj, detrended_obs, detrended_sim,
                    empirical_moments, sim_moments, lookback, output_path):
    """
    Crea figura di confronto con:
    - Top left: prezzi osservati vs simulati
    - Top right: detrended osservati vs simulati
    - Bottom left: distribuzione log-returns
    - Bottom right: tabella momenti
    """
    fig = plt.figure(figsize=(14, 10))

    # Setup days
    days_obs = np.arange(len(prices))
    days_sim = np.arange(lookback, len(prices))

    # 1. Prezzi
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(days_obs, prices, 'b-', linewidth=1.5, label='Observed', alpha=0.9)
    ax1.plot(days_sim, representative_traj, 'r-', linewidth=1.2, label='Simulated', alpha=0.8)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title(f'{commodity.upper()} - Price Trajectories')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Detrended
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(days_obs, detrended_obs, 'b-', linewidth=1.5, label='Observed', alpha=0.9)
    ax2.plot(days_sim, detrended_sim, 'r-', linewidth=1.2, label='Simulated', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Detrended log-price')
    ax2.set_title(f'{commodity.upper()} - Detrended (Mean-Reverting)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. Distribuzione log-returns
    ax3 = fig.add_subplot(2, 2, 3)
    obs_returns = np.diff(np.log(prices))
    sim_returns = np.diff(np.log(representative_traj))

    # Istogrammi sovrapposti
    bins = np.linspace(min(obs_returns.min(), sim_returns.min()),
                       max(obs_returns.max(), sim_returns.max()), 50)
    ax3.hist(obs_returns, bins=bins, alpha=0.5, color='blue', label='Observed', density=True)
    ax3.hist(sim_returns, bins=bins, alpha=0.5, color='red', label='Simulated', density=True)
    ax3.set_xlabel('Log-return')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Log-Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Tabella momenti
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Calcola errori
    std_err = (sim_moments['std'] - empirical_moments['std']) / empirical_moments['std'] * 100
    kurt_err = (sim_moments['kurtosis'] - empirical_moments['kurtosis']) / empirical_moments['kurtosis'] * 100

    table_data = [
        ['Moment', 'Observed', 'Simulated', 'Error'],
        ['Mean', f"{empirical_moments['mean']:.6f}", f"{sim_moments['mean']:.6f}", '-'],
        ['Std', f"{empirical_moments['std']:.6f}", f"{sim_moments['std']:.6f}", f"{std_err:+.1f}%"],
        ['Skewness', f"{empirical_moments['skewness']:.4f}", f"{sim_moments['skewness']:.4f}",
         f"{sim_moments['skewness'] - empirical_moments['skewness']:+.2f}"],
        ['Kurtosis', f"{empirical_moments['kurtosis']:.2f}", f"{sim_moments['kurtosis']:.2f}", f"{kurt_err:+.1f}%"],
    ]

    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Moments Comparison (Single Trajectory)', pad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 65)
    print("  REPRESENTATIVE TRAJECTORY SELECTION")
    print("  Matching empirical moments")
    print("=" * 65)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    print("\n  Loading price data...")
    gas_prices = load_prices('data/gas_1826.dat')
    power_prices = load_prices('data/power_1826.dat')
    print(f"    Gas: {len(gas_prices)} observations")
    print(f"    Power: {len(power_prices)} observations")

    commodities = [
        ('gas', gas_prices),
        ('power', power_prices),
    ]

    results = {}

    for commodity, prices in commodities:
        print(f"\n{'-'*65}")
        print(f"  Processing {commodity.upper()}")
        print(f"{'-'*65}")

        # Momenti empirici
        empirical_returns = np.diff(np.log(prices))
        empirical_moments = compute_moments(empirical_returns)

        print(f"\n  Empirical moments:")
        print(f"    Std:      {empirical_moments['std']:.6f}")
        print(f"    Skewness: {empirical_moments['skewness']:.4f}")
        print(f"    Kurtosis: {empirical_moments['kurtosis']:.2f}")

        # Carica modello
        print(f"\n  Loading model...")
        try:
            model, config, source = load_model(commodity)
            print(f"    Config: lookback={config['lookback']}, n_components={config['n_components']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Detrend
        log_prices, trend, detrended = loess_detrend(prices, frac=config['loess_frac'])

        # Genera traiettorie
        print(f"\n  Generating {N_TRAJECTORIES} trajectories...")
        price_traj, detrended_traj = generate_price_trajectories(
            model, detrended, trend, config, N_TRAJECTORIES
        )
        print(f"    Shape: {price_traj.shape}")

        # Seleziona rappresentativa (considera momenti + picchi estremi)
        print(f"\n  Selecting representative trajectory...")
        print(f"    Observed detrended range: [{detrended.min():.3f}, {detrended.max():.3f}]")
        best_idx, best_moments, best_score = select_representative_trajectory(
            price_traj, empirical_moments, prices, detrended_traj, detrended
        )

        print(f"\n  Best trajectory: #{best_idx} (score: {best_score:.4f})")
        print(f"    Simulated moments:")
        print(f"      Std:      {best_moments['std']:.6f} ({(best_moments['std']/empirical_moments['std']-1)*100:+.1f}%)")
        print(f"      Skewness: {best_moments['skewness']:.4f} ({best_moments['skewness']-empirical_moments['skewness']:+.2f})")
        print(f"      Kurtosis: {best_moments['kurtosis']:.2f} ({(best_moments['kurtosis']/empirical_moments['kurtosis']-1)*100:+.1f}%)")
        print(f"    Detrended range: [{best_moments['min']:.3f}, {best_moments['max']:.3f}]")

        # Plot
        lookback = config['lookback']
        output_path = f'representative_{commodity}.png'

        plot_comparison(
            commodity=commodity,
            prices=prices,
            representative_traj=price_traj[best_idx],
            detrended_obs=detrended,
            detrended_sim=detrended_traj[best_idx],
            empirical_moments=empirical_moments,
            sim_moments=best_moments,
            lookback=lookback,
            output_path=output_path
        )

        results[commodity] = {
            'empirical': empirical_moments,
            'simulated': best_moments,
            'trajectory_idx': best_idx,
            'score': best_score
        }

    # Riepilogo finale
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    for commodity in ['gas', 'power']:
        if commodity not in results:
            continue
        r = results[commodity]
        std_err = (r['simulated']['std'] / r['empirical']['std'] - 1) * 100
        kurt_err = (r['simulated']['kurtosis'] / r['empirical']['kurtosis'] - 1) * 100
        print(f"\n  {commodity.upper()}:")
        print(f"    Std error:      {std_err:+.1f}%")
        print(f"    Kurtosis error: {kurt_err:+.1f}%")
        print(f"    Output: representative_{commodity}.png")

    print("\n" + "=" * 65)
    print("  COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
