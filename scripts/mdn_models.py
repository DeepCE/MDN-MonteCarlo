#!/usr/bin/env python3
"""Classi modelli MDN condivise (base + Enhanced) e helpers di loading.

Importabile da training / generation / analysis scripts per evitare duplicazione.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AutoregressiveMDN(nn.Module):
    """MDN base: LSTM + 3 heads FC (pi, mu, sigma). Usata in PJM/WTI ri-addestrati."""

    def __init__(self, lookback=30, hidden_dim=128, n_layers=2, n_components=8,
                 dropout=0.15):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_components = n_components
        self.dropout = dropout
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


class AutoregressiveMDNLegacy(nn.Module):
    """MDN originale del paper: LSTM + fc_hidden(Linear+ReLU+Dropout) + 3 heads.

    Struttura riprodotta da scripts/train_mdn_final.py usato per i ckpt paper:
    mdn_final_gas.pt, mdn_fulldata_gas.pt, mdn_gas_tuned.pt, mdn_gas_augmented.pt.
    Nota: sigma = softplus(.) + 1e-4 (non 1e-6 come in AutoregressiveMDN).
    """

    def __init__(self, lookback=30, hidden_dim=128, n_layers=2, n_components=5,
                 dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_components = n_components
        self.dropout = dropout
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        h = self.fc_hidden(h)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4
        return pi, mu, sigma


class EnhancedMDN(nn.Module):
    """MDN con residual FC blocks + LayerNorm + GELU + pre-MDN projection.

    Configurazione del paper originale per PUN (power IT).
    Architettura:
      LSTM encoder -> [FC-GELU-Dropout-FC + residual + LayerNorm] * n_hidden_layers
                   -> FC-GELU-Dropout (pre-MDN)
                   -> 3 heads (pi, mu, sigma)
    """

    def __init__(self, lookback=30, hidden_dim=96, n_layers=2, n_components=8,
                 n_hidden_layers=2, dropout=0.15):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_components = n_components
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_hidden_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                          for _ in range(n_hidden_layers)])

        self.pre_mdn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)

        # Init bias of sigma toward small positive (paper original trick)
        nn.init.constant_(self.fc_sigma.bias, -1.0)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            residual = h
            h = layer(h)
            h = norm(h + residual)
        h = self.pre_mdn(h)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4
        return pi, mu, sigma


class EnhancedMDNt(nn.Module):
    """MDN with Student-t mixture output head.

    Same architecture as EnhancedMDN but with an extra per-component degrees-of-freedom
    head (nu). Each component is Student-t(nu_k, mu_k, sigma_k). The family of
    mixtures of Student-t is universal in L^1 (like Gaussian mixtures) and is a
    strict superset of Gaussian mixtures (limit nu -> inf recovers Gaussian). The
    per-component nu gives individual polynomial tail-decay control, removing the
    finite-K kurtosis ceiling of Gaussian mixtures.
    """

    def __init__(self, lookback=20, hidden_dim=128, n_layers=2, n_components=5,
                 n_hidden_layers=2, dropout=0.15):
        super().__init__()
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_components = n_components
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_hidden_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                          for _ in range(n_hidden_layers)])

        self.pre_mdn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)
        self.fc_nu = nn.Linear(hidden_dim, n_components)

        nn.init.constant_(self.fc_sigma.bias, -1.0)
        # Initialize nu bias so softplus(bias) + 2.1 ≈ 5 (moderate heavy-tail prior)
        nn.init.constant_(self.fc_nu.bias, 1.055)

    def forward(self, x):
        h = self.lstm(x)[0][:, -1, :]
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            residual = h
            h = layer(h)
            h = norm(h + residual)
        h = self.pre_mdn(h)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)) + 1e-4
        nu = torch.nn.functional.softplus(self.fc_nu(h)) + 2.1
        return pi, mu, sigma, nu


def mdn_t_loss(pi, mu, sigma, nu, target):
    """Negative log-likelihood of a Student-t mixture.

    log f_t(x; nu, mu, sigma) =
        lgamma((nu+1)/2) - lgamma(nu/2)
        - 0.5 * log(nu * pi) - log(sigma)
        - ((nu+1)/2) * log(1 + ((x-mu)/sigma)^2 / nu)
    """
    target = target.unsqueeze(-1)
    z = (target - mu) / sigma
    log_const = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
                 - 0.5 * torch.log(nu * np.pi) - torch.log(sigma))
    log_kernel = -((nu + 1) / 2) * torch.log1p(z ** 2 / nu)
    log_prob = log_const + log_kernel
    log_pi = torch.log(pi + 1e-10)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


def _detect_arch(state_dict, cfg_arch: str | None) -> str:
    """Inferisce l'architettura dai tensori nel state_dict.

    Prevale il state_dict sul campo cfg['arch'] (i ckpt paper hanno config incompleti).
    """
    keys = set(state_dict.keys())
    has_nu = any(k.startswith("fc_nu.") for k in keys)
    has_enhanced = (any(k.startswith("hidden_layers.") for k in keys) or
                    any(k.startswith("layer_norms.") for k in keys) or
                    any(k.startswith("pre_mdn.") for k in keys))
    if has_enhanced and has_nu:
        return "enhanced_t"
    if has_enhanced:
        return "enhanced"
    if any(k.startswith("fc_hidden.") for k in keys):
        return "legacy"
    if cfg_arch in {"base", "enhanced", "legacy"}:
        return cfg_arch
    return "base"


def build_model(cfg: dict, arch: str | None = None) -> nn.Module:
    """Costruisce il modello in base all'arch ('base', 'enhanced', 'legacy').

    Se arch e' None usa cfg['arch'] con default 'base'.
    """
    if arch is None:
        arch = cfg.get("arch", "base")
    if arch == "enhanced_t":
        return EnhancedMDNt(
            lookback=cfg["lookback"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            n_components=cfg["n_components"],
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            dropout=cfg.get("dropout", 0.15),
        )
    if arch == "enhanced":
        return EnhancedMDN(
            lookback=cfg["lookback"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            n_components=cfg["n_components"],
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            dropout=cfg.get("dropout", 0.15),
        )
    if arch == "legacy":
        return AutoregressiveMDNLegacy(
            lookback=cfg["lookback"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            n_components=cfg["n_components"],
            dropout=cfg.get("dropout", 0.1),
        )
    return AutoregressiveMDN(
        lookback=cfg["lookback"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_components=cfg["n_components"],
        dropout=cfg.get("dropout", 0.15),
    )


def load_checkpoint_model(ckpt_path, device):
    """Carica checkpoint + costruisce il modello corretto ispezionando lo state_dict.

    I ckpt paper originali hanno config['arch'] mancante: l'arch viene dedotta
    dai tensor keys del state_dict e prevale sul config.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["model_state_dict"]
    arch = _detect_arch(sd, cfg.get("arch"))
    model = build_model(cfg, arch=arch).to(device)
    model.load_state_dict(sd)
    model.eval()
    ckpt["_detected_arch"] = arch
    return model, ckpt


def mdn_loss(pi, mu, sigma, target):
    target = target.unsqueeze(-1)
    var = sigma ** 2
    log_prob = -0.5 * ((target - mu) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
    log_pi = torch.log(pi + 1e-10)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()
