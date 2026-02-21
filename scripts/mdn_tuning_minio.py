#!/usr/bin/env python3
"""
MDN Hyperparameter Tuning - legge dati da MinIO.
Eseguire sul cluster Ray con: ray job submit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.fs as pafs

from ray import tune
from ray.tune.schedulers import ASHAScheduler

# ============================================
# DATA LOADING DA MINIO
# ============================================

def load_prices_from_minio(commodity):
    """Carica prezzi da MinIO S3."""
    fs = pafs.S3FileSystem(endpoint_override="http://YOUR_MINIO_HOST:9000")
    path = f"ray-cluster/datasets/{commodity}_1826.dat"

    with fs.open_input_file(path) as f:
        content = f.read().decode('utf-8')

    prices = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            prices.append(float(line.replace(',', '.')))
    return np.array(prices)


def loess_detrend(prices, frac=0.1):
    """Detrending con kernel gaussiano."""
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
    return log_prices - trend


# ============================================
# MODEL
# ============================================

class AutoregressiveMDN(nn.Module):
    """LSTM + Mixture Density Network."""

    def __init__(self, lookback=20, hidden_dim=64, n_layers=2, n_components=5, dropout=0.0):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-6
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, target):
    """Negative log-likelihood per MDN."""
    target = target.unsqueeze(-1)
    var = sigma ** 2
    log_prob = -0.5 * ((target - mu) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
    log_pi = torch.log(pi + 1e-10)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


# ============================================
# TRAINING FUNCTION
# ============================================

def train_mdn(config):
    """Funzione di training per Ray Tune."""

    # Carica dati da MinIO
    commodity = config["commodity"]
    prices = load_prices_from_minio(commodity)
    detrended = loess_detrend(prices, frac=0.1)

    # Prepara sequenze
    lookback = config["lookback"]
    X = np.array([detrended[i:i+lookback] for i in range(len(detrended)-lookback)])
    y = np.array([detrended[i+lookback] for i in range(len(detrended)-lookback)])

    # Train/val split
    split = int(0.8 * len(X))
    X_train = torch.FloatTensor(X[:split]).unsqueeze(-1)
    y_train = torch.FloatTensor(y[:split])
    X_val = torch.FloatTensor(X[split:]).unsqueeze(-1)
    y_val = torch.FloatTensor(y[split:])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    loader = DataLoader(TensorDataset(X_train, y_train),
                       batch_size=config["batch_size"], shuffle=True)

    # Modello
    model = AutoregressiveMDN(
        lookback=config["lookback"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_components=config["n_components"],
        dropout=config.get("dropout", 0.0)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config.get("patience", 20)

    for epoch in range(config["max_epochs"]):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mdn_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = mdn_loss(*model(X_val), y_val).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Report metrics to Ray Tune (dict as first positional arg)
        tune.report({"train_loss": train_loss, "val_loss": val_loss,
                     "best_val_loss": best_val_loss, "epoch": epoch})

        if patience_counter >= patience:
            break


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MDN HYPERPARAMETER TUNING - RAY CLUSTER")
    print("=" * 60)

    # Search space
    search_space = {
        "commodity": tune.grid_search(["gas", "power"]),
        "lookback": tune.grid_search([20, 30]),
        "hidden_dim": tune.grid_search([64, 128]),
        "n_layers": tune.grid_search([2, 3]),
        "n_components": tune.grid_search([5, 8]),
        "dropout": tune.sample_from(lambda spec: 0.1 if spec.config.n_layers > 2 else 0.0),
        "lr": 0.001,
        "batch_size": 256,  # Aumentato per migliore utilizzo GPU
        "max_epochs": 150,
        "patience": 25
    }

    # ASHA scheduler per early stopping
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=150,
        grace_period=30,
        reduction_factor=2
    )

    # 2 commodity × 2 lookback × 2 hidden × 2 layers × 2 components = 32 trials
    print(f"\n  Search space: 32 configurazioni")
    print(f"  GPU: 3 (parallelo)")
    print(f"  Dati: s3://ray-cluster/datasets/")

    # Esegui tuning
    analysis = tune.run(
        train_mdn,
        config=search_space,
        resources_per_trial={"gpu": 1, "cpu": 2},
        num_samples=1,
        scheduler=scheduler,
        storage_path="s3://ray-cluster/runs/",
        name="mdn-hyperparameter-search",
        verbose=1,
        max_concurrent_trials=3,
    )

    # Risultati
    print("\n" + "=" * 60)
    print("  RISULTATI")
    print("=" * 60)

    # Usa best_result (proprieta') e best_config
    try:
        best_config = analysis.best_config
        best_df = analysis.results_df
        best_val = best_df["best_val_loss"].min()
        print(f"\n  BEST OVERALL:")
        print(f"    val_loss: {best_val:.4f}")
        print(f"    config: {best_config}")
    except Exception as e:
        print(f"\n  Errore best_result: {e}")

    # Stampa anche i migliori per ogni commodity usando dataframe
    try:
        df = analysis.results_df
        print(f"\n  Colonne disponibili: {list(df.columns)[:10]}...")  # Debug

        # Trova il nome corretto della colonna commodity
        commodity_col = None
        for col in df.columns:
            if 'commodity' in col.lower():
                commodity_col = col
                break

        if commodity_col:
            for commodity in ["gas", "power"]:
                df_c = df[df[commodity_col] == commodity]
                if len(df_c) > 0:
                    # Trova la colonna val_loss
                    val_col = "val_loss" if "val_loss" in df.columns else "best_val_loss"
                    best_idx = df_c[val_col].idxmin()
                    best = df_c.loc[best_idx]
                    print(f"\n  BEST {commodity.upper()}:")
                    print(f"    val_loss: {best.get(val_col, 'N/A')}")
                    for key in best.index:
                        if 'config' in key.lower() or 'lookback' in key or 'hidden' in key:
                            print(f"    {key}: {best[key]}")
    except Exception as e:
        print(f"\n  Errore parsing risultati: {e}")
        print("  Controlla i risultati su MinIO o dashboard")

    print(f"\n  Risultati: s3://ray-cluster/runs/mdn-hyperparameter-search/")
    print("=" * 60)
