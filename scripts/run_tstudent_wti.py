#!/usr/bin/env python3
"""Component modularity stress-test on WTI: Student-t mixture MDN.

Rationale (Section 5.4, Framework Modularity — component axis):
The default framework uses a Gaussian mixture output head. Replacing Gaussian
components with Student-t components preserves the universal-approximator
property of the mixture family while giving each component a per-component
polynomial tail decay (parameter nu). On WTI, with empirical kurtosis > 30, this
is the natural extension: individual-component kurtosis ceiling of Gaussian
components is expected to be the binding constraint. Architecture, hyperparameters,
training protocol, and evaluation pipeline are unchanged; only the output
component family changes.

This single script:
  1. Trains EnhancedMDNt on WTI with the same config used for the Gaussian MDN
     (Phase 1, 85/15 chronological split, patience 25).
  2. Runs the one-day-ahead VaR backtest (Kupiec + Christoffersen) at
     alpha in {0.01, 0.05} on both tails.
  3. Runs the PIT diagnostic suite (KS, Anderson-Darling, Ljung-Box,
     Berkowitz) on u_t = F_{tmix(t)}(xi_{t+1}^emp).
  4. Generates 5000 Monte Carlo trajectories and reports coverage of empirical
     ACF at lags 1..60 for both xi_t and r_t^2.

Output: data/tstudent_wti_summary.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as spstats
from scipy.stats import norm, t as tdist, chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from mdn_models import build_model, mdn_t_loss  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"

CFG = dict(arch="enhanced_t", lookback=20, hidden_dim=128, n_layers=2,
           n_components=5, dropout=0.15, n_hidden_layers=2)

TRAIN_FRAC = 0.85
MAX_EPOCHS = 200
PATIENCE = 25
LR = 1e-3
BATCH_SIZE = 256
SEED = 42
N_PATHS = 5000
MAX_LAG = 60
ALPHAS = (0.01, 0.05)


def load_dat(path: Path) -> np.ndarray:
    with open(path) as f:
        return np.array([float(line.strip().replace(",", "."))
                         for line in f if line.strip()])


def loess_detrend(prices, frac=0.1):
    lp = np.log(prices)
    n = len(lp)
    w = max(5, int(n * frac) | 1)
    hw = w // 2
    wts = np.exp(-np.arange(-hw, hw + 1) ** 2 / (2 * (hw / 2) ** 2))
    wts /= wts.sum()
    pad = np.pad(lp, hw, mode="edge")
    trend = np.array([np.sum(pad[i:i + w] * wts) for i in range(n)])
    return lp - trend


def make_split(xi, lookback):
    n = len(xi) - lookback
    X = np.stack([xi[i:i + lookback] for i in range(n)])
    y = xi[lookback:]
    split = int(n * TRAIN_FRAC)
    return X[:split], y[:split], X[split:], y[split:]


# ---------- Training ----------
def train_wti(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)

    xi = loess_detrend(load_dat(DATA / "wti_2501.dat"))
    X_tr, y_tr, X_va, y_va = make_split(xi, CFG["lookback"])
    print(f"[wti] train={len(y_tr)} val={len(y_va)} T={len(xi)}", flush=True)

    Xt = torch.FloatTensor(X_tr).unsqueeze(-1).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_va).unsqueeze(-1).to(device)
    yv = torch.FloatTensor(y_va).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(CFG, arch=CFG["arch"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: EnhancedMDNt K={CFG['n_components']} params={n_params}", flush=True)
    opt = optim.Adam(model.parameters(), lr=LR)

    best, best_state, counter, best_ep = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = mdn_t_loss(*model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = mdn_t_loss(*model(Xv), yv).item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep {epoch + 1:3d}  val NLL = {vl:+.4f}", flush=True)
        if vl < best:
            best, counter, best_ep = vl, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        if counter >= PATIENCE:
            print(f"  early stop ep {epoch + 1}  best NLL={best:+.4f} at ep {best_ep}", flush=True)
            break
    model.load_state_dict(best_state)
    print(f"  training done in {time.time() - t0:.1f}s", flush=True)

    # Save checkpoint
    MODELS.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS / "mdn_v2_wti_tstudent.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {**CFG},
        "market": "wti",
        "phase1_best_val": float(best),
        "initial_history": xi[-CFG["lookback"]:].astype(np.float32),
        "lookback": CFG["lookback"],
    }, ckpt_path)
    print(f"  saved: {ckpt_path.name}", flush=True)
    return model, xi, best


# ---------- Forward-pass all dates ----------
def forward_all(model, xi, device):
    m = CFG["lookback"]
    T_eff = len(xi) - m
    X = np.stack([xi[t:t + m] for t in range(T_eff)], axis=0).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pis, mus, sigmas, nus = [], [], [], []
        for s in range(0, T_eff, 512):
            p, mu, sg, nu = model(X_t[s:s + 512])
            pis.append(p.cpu().numpy()); mus.append(mu.cpu().numpy())
            sigmas.append(sg.cpu().numpy()); nus.append(nu.cpu().numpy())
    return (np.concatenate(pis), np.concatenate(mus),
            np.concatenate(sigmas), np.concatenate(nus))


# ---------- VaR ----------
def tmix_cdf(pi, mu, sigma, nu, x):
    return float(np.sum(pi * tdist.cdf(x, nu, loc=mu, scale=sigma)))


def tmix_quantile(pi, mu, sigma, nu, q, grid=2048):
    lo = float((mu - 12 * sigma).min())
    hi = float((mu + 12 * sigma).max())
    xs = np.linspace(lo, hi, grid)
    cdf = np.zeros_like(xs)
    for k in range(len(pi)):
        cdf += pi[k] * tdist.cdf(xs, nu[k], loc=mu[k], scale=sigma[k])
    idx = int(np.clip(np.searchsorted(cdf, q), 1, grid - 1))
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    return float(x0) if y1 == y0 else float(x0 + (q - y0) / (y1 - y0) * (x1 - x0))


def kupiec_lr(hits, alpha):
    n = len(hits); x = int(hits.sum())
    phat = x / n if n > 0 else 0.0
    if phat <= 0 or phat >= 1:
        return 0.0, 1.0
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll1 = x * np.log(phat) + (n - x) * np.log(1 - phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_ind(hits):
    hits = hits.astype(int)
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return 0.0, 1.0
    p01 = n01 / n0; p11 = n11 / n1
    p_any = (n01 + n11) / (n0 + n1)
    if any(x <= 0 or x >= 1 for x in [p_any, p01, p11]):
        return 0.0, 1.0
    ll0 = (n00 + n10) * np.log(1 - p_any) + (n01 + n11) * np.log(p_any)
    ll1 = (n00 * np.log(1 - p01) + n01 * np.log(p01)
           + n10 * np.log(1 - p11) + n11 * np.log(p11))
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def run_var(PI, MU, SIGMA, NU, xi):
    m = CFG["lookback"]
    T_eff = len(xi) - m
    r_emp = xi[m:] - xi[m - 1:-1]
    results = {}
    for alpha in ALPHAS:
        var_l = np.empty(T_eff); var_r = np.empty(T_eff)
        for t in range(T_eff):
            q_lo = tmix_quantile(PI[t], MU[t], SIGMA[t], NU[t], alpha)
            q_hi = tmix_quantile(PI[t], MU[t], SIGMA[t], NU[t], 1 - alpha)
            xi_cur = float(xi[m - 1 + t])
            var_l[t] = q_lo - xi_cur
            var_r[t] = q_hi - xi_cur
        hl = (r_emp < var_l).astype(int)
        hr = (r_emp > var_r).astype(int)
        for tail, h in [("left", hl), ("right", hr)]:
            n = len(h); x = int(h.sum())
            lr_uc, p_uc = kupiec_lr(h, alpha)
            lr_ind, p_ind = christoffersen_ind(h)
            lr_cc = lr_uc + lr_ind
            p_cc = float(1 - chi2.cdf(lr_cc, df=2))
            results[f"alpha={alpha:.2f}_{tail}"] = dict(
                n=n, exceed=x, rate=x / n, expected=alpha,
                lr_uc=lr_uc, p_uc=p_uc, lr_ind=lr_ind, p_ind=p_ind,
                lr_cc=lr_cc, p_cc=p_cc)
    return results


# ---------- PIT ----------
def ljung_box(x, lag=10):
    x = np.asarray(x) - np.mean(x)
    n = len(x); g0 = np.sum(x * x) / n; q = 0.0
    for k in range(1, lag + 1):
        gk = np.sum(x[:-k] * x[k:]) / n
        q += (gk / g0) ** 2 / (n - k)
    q *= n * (n + 2)
    return float(q), float(1 - chi2.cdf(q, df=lag))


def berkowitz(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    z = norm.ppf(u); n = len(z)
    ll_null = np.sum(norm.logpdf(z, 0, 1))
    X = np.column_stack([np.ones(n - 1), z[:-1]])
    y = z[1:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, rho = beta
    resid = y - X @ beta; s2 = np.mean(resid ** 2)
    ll_alt = np.sum(norm.logpdf(y, X @ beta, np.sqrt(s2)))
    ll_alt += norm.logpdf(z[0], a / (1 - rho + 1e-9),
                          np.sqrt(s2 / max(1 - rho ** 2, 1e-6)))
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - chi2.cdf(lr, df=3)), float(a), float(rho), float(np.sqrt(s2))


def run_pit(PI, MU, SIGMA, NU, xi):
    m = CFG["lookback"]
    T_eff = len(xi) - m
    target = xi[m:]
    u = np.empty(T_eff)
    for t in range(T_eff):
        u[t] = tmix_cdf(PI[t], MU[t], SIGMA[t], NU[t], float(target[t]))
    ks_s, ks_p = spstats.kstest(u, "uniform")
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    ad = spstats.anderson(z, dist="norm")
    ad_s = float(ad.statistic)
    lvls = [0.15, 0.10, 0.05, 0.025, 0.01]
    if ad_s < ad.critical_values[0]:
        ad_p = "> 0.15"
    elif ad_s >= ad.critical_values[-1]:
        ad_p = "< 0.01"
    else:
        ln_p = np.interp(ad_s, ad.critical_values, np.log(lvls))
        ad_p = f"~{np.exp(ln_p):.3f}"
    lb_s, lb_p = ljung_box(u - 0.5, 10)
    bk_s, bk_p, bk_mu, bk_rho, bk_sig = berkowitz(u)
    return dict(n=T_eff, u_mean=float(u.mean()), u_std=float(u.std(ddof=1)),
                ks=(ks_s, ks_p), ad=(ad_s, ad_p), lb=(lb_s, lb_p),
                bk=(bk_s, bk_p, bk_mu, bk_rho, bk_sig))


# ---------- MC paths + ACF coverage ----------
def sample_tmix(pi, mu, sigma, nu, rng):
    """Sample one observation from t-Student mixture [K] arrays."""
    k = rng.choice(len(pi), p=pi)
    return mu[k] + sigma[k] * rng.standard_t(nu[k])


def generate_paths(model, xi, device, n_paths=N_PATHS, seed=123):
    m = CFG["lookback"]
    T = len(xi)
    rng = np.random.default_rng(seed)
    # Initialize all paths with the last m observed values
    paths = np.zeros((n_paths, T), dtype=np.float32)
    paths[:, :m] = xi[:m]
    contexts = torch.FloatTensor(np.tile(xi[:m], (n_paths, 1))).unsqueeze(-1).to(device)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for t in range(m, T):
            pi, mu, sigma, nu = model(contexts)
            pi = pi.cpu().numpy(); mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy(); nu = nu.cpu().numpy()
            step = np.array([sample_tmix(pi[i], mu[i], sigma[i], nu[i], rng)
                             for i in range(n_paths)], dtype=np.float32)
            paths[:, t] = step
            # Slide context window
            new_ctx = np.column_stack([
                contexts.cpu().numpy().squeeze(-1)[:, 1:],
                step[:, None]
            ])
            contexts = torch.from_numpy(new_ctx).unsqueeze(-1).to(device)
            if (t - m + 1) % 500 == 0:
                print(f"  gen step {t - m + 1}/{T - m}  elapsed={time.time() - t0:.1f}s",
                      flush=True)
    print(f"  MC generation done in {time.time() - t0:.1f}s", flush=True)
    return paths


def acf_single(x, maxlag):
    x = x - x.mean()
    var = (x ** 2).mean()
    if var <= 0:
        return np.zeros(maxlag)
    return np.array([(x[:-k] * x[k:]).mean() / var for k in range(1, maxlag + 1)])


def acf_coverage(emp_series, sim_paths, maxlag=MAX_LAG):
    emp_acf = acf_single(emp_series, maxlag)
    sim_acfs = np.stack([acf_single(sim_paths[i], maxlag) for i in range(len(sim_paths))])
    lo = np.percentile(sim_acfs, 2.5, axis=0)
    hi = np.percentile(sim_acfs, 97.5, axis=0)
    in_band = (emp_acf >= lo) & (emp_acf <= hi)
    return float(np.mean(in_band)), emp_acf, lo, hi


# ---------- Main ----------
def main():
    device = torch.device("cpu")
    print(f"device: {device}  SEED: {SEED}", flush=True)
    print("=" * 70 + "\n  COMPONENT MODULARITY -- WTI / Student-t mixture\n" + "=" * 70,
          flush=True)

    model, xi, best_val = train_wti(device)
    PI, MU, SIGMA, NU = forward_all(model, xi, device)

    print("\n--- VaR backtest ---", flush=True)
    var_res = run_var(PI, MU, SIGMA, NU, xi)
    print("\n--- PIT diagnostics ---", flush=True)
    pit_res = run_pit(PI, MU, SIGMA, NU, xi)
    print("\n--- MC paths + ACF coverage ---", flush=True)
    paths = generate_paths(model, xi, device, n_paths=N_PATHS, seed=123)
    cov_xi, *_ = acf_coverage(xi, paths, MAX_LAG)
    r_emp = np.diff(xi)
    r_sim = np.diff(paths, axis=1)
    cov_r2, *_ = acf_coverage(r_emp ** 2, r_sim ** 2, MAX_LAG)

    # ----------------- summary -----------------
    lines = []
    lines.append("=" * 104)
    lines.append("  WTI -- Component modularity: Student-t mixture MDN")
    lines.append(f"  Config: {CFG}")
    lines.append(f"  Phase 1 best val NLL = {best_val:+.4f}  (ref Gaussian MDN: -2.52)")
    lines.append("=" * 104)
    lines.append("\n[VaR backtest]")
    lines.append(f"  {'level':<6} {'tail':<6} {'N':>5} {'exc':>5} {'rate':>7} {'exp':>5} "
                 f"{'LR_uc':>7} {'p_uc':>7} {'LR_ind':>7} {'p_ind':>7} {'LR_cc':>7} {'p_cc':>7}")
    for key, r in var_res.items():
        lvl, tail = key.split("_")
        lines.append(
            f"  {lvl:<6} {tail:<6} {r['n']:>5d} {r['exceed']:>5d} {r['rate']:>7.4f} "
            f"{r['expected']:>5.2f} {r['lr_uc']:>7.3f} {r['p_uc']:>7.4f} "
            f"{r['lr_ind']:>7.3f} {r['p_ind']:>7.4f} {r['lr_cc']:>7.3f} {r['p_cc']:>7.4f}"
        )
    lines.append("\n[PIT diagnostics]")
    ks_s, ks_p = pit_res["ks"]; ad_s, ad_p = pit_res["ad"]
    lb_s, lb_p = pit_res["lb"]; bk_s, bk_p, bk_mu, bk_rho, bk_sig = pit_res["bk"]
    lines.append(f"  N={pit_res['n']}  u_mean={pit_res['u_mean']:.4f}  "
                 f"u_std={pit_res['u_std']:.4f}")
    lines.append(f"  KS  stat={ks_s:.4f}  p={ks_p:.4f}")
    lines.append(f"  AD  stat={ad_s:.3f}  p={ad_p}")
    lines.append(f"  LB  stat={lb_s:.2f}   p={lb_p:.4f}")
    lines.append(f"  Berk LR={bk_s:.2f}  p={bk_p:.4f}   "
                 f"mu={bk_mu:+.3f}  rho={bk_rho:+.3f}  sigma={bk_sig:.3f}")
    lines.append("\n[ACF coverage (per-path 95% bands, 60 lags)]")
    lines.append(f"  coverage rho_l(xi_t)   = {cov_xi:.0%}")
    lines.append(f"  coverage rho_l(r_t^2)  = {cov_r2:.0%}")
    lines.append("\n[Gaussian MDN reference (from var_backtest_summary.txt / pit_summary.txt)]")
    lines.append("  VaR: 1/4 pass conditional coverage; u_mean=0.58, sigma_B=0.87")
    lines.append("  ACF: xi=58%, r^2=43%")
    lines.append("=" * 104)

    text = "\n".join(lines)
    print("\n" + text)
    out = DATA / "tstudent_wti_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
