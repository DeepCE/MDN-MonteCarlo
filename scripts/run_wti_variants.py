#!/usr/bin/env python3
"""Two WTI variants in the Gaussian MDN family.

MODE "moment":
  Moment-weighted NLL training. The loss is L = NLL + lambda * (kurt_pred - kurt_emp)^2 / kurt_emp^2.
  Higher-order central moments of the conditional GMM are computed in closed form
  (Bishop 1994, Titterington 1985) and aggregated across the training batch.
  Frames as the "moment axis" of framework modularity (Section 5.4).

MODE "cushing":
  Diagnostic experiment. Training data excludes the Cushing Apr 15 -- May 4, 2020
  window (empirical xi indices 1072..1090). Evaluation is on the full series.
  Tests the hypothesis that the WTI VaR rejections reflect training contamination
  from the 2020 oil dislocation rather than a structural model limitation.

Both experiments use the baseline Gaussian MDN architecture and config. Only
the training loss (moment) or training data (cushing) changes.

Output:  data/wti_variant_{mode}_summary.txt
Checkpoint:  models/mdn_v2_wti_{mode}.pt

Usage:
  python code/run_wti_variants.py --mode moment   --lambda 0.1
  python code/run_wti_variants.py --mode cushing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as spstats
from scipy.stats import norm, chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from mdn_models import build_model, mdn_loss  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"

CFG = dict(arch="enhanced", lookback=20, hidden_dim=128, n_layers=2,
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
CUSHING_WINDOW = (1072, 1090)  # empirical xi indices


def load_dat(p: Path) -> np.ndarray:
    with open(p) as f:
        return np.array([float(l.strip().replace(",", "."))
                         for l in f if l.strip()])


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


# ---------------- Moment machinery ----------------
def gmm_conditional_central_moments(pi, mu, sigma):
    """pi, mu, sigma: [B, K]. Returns per-sample central moments of the GMM."""
    cond_mean = (pi * mu).sum(dim=-1)                        # [B]
    delta = mu - cond_mean.unsqueeze(-1)                     # [B, K]
    m2 = (pi * (sigma ** 2 + delta ** 2)).sum(dim=-1)
    m3 = (pi * (3 * sigma ** 2 * delta + delta ** 3)).sum(dim=-1)
    m4 = (pi * (3 * sigma ** 4 + 6 * sigma ** 2 * delta ** 2 + delta ** 4)).sum(dim=-1)
    return cond_mean, m2, m3, m4


def marginal_moments_from_conditional(cm, m2, m3, m4):
    """Aggregate conditional central moments to marginal central moments.
    Law of total central moments applied across the batch.
    """
    mu_marg = cm.mean()
    d = cm - mu_marg
    var_marg = m2.mean() + (d ** 2).mean()
    m3_marg = m3.mean() + 3 * (m2 * d).mean() + (d ** 3).mean()
    m4_marg = (m4.mean() + 4 * (m3 * d).mean()
               + 6 * (m2 * d ** 2).mean() + (d ** 4).mean())
    return mu_marg, var_marg, m3_marg, m4_marg


def moment_penalty_kurt(pi, mu, sigma, kurt_emp):
    cm, m2, m3, m4 = gmm_conditional_central_moments(pi, mu, sigma)
    _, var_marg, _, m4_marg = marginal_moments_from_conditional(cm, m2, m3, m4)
    kurt_pred = m4_marg / (var_marg ** 2 + 1e-10)
    return ((kurt_pred - kurt_emp) / (kurt_emp + 1e-10)) ** 2, kurt_pred


# ---------------- Training ----------------
def make_sequences(xi, lookback):
    n = len(xi) - lookback
    X = np.stack([xi[i:i + lookback] for i in range(n)])
    y = xi[lookback:]
    return X, y


def train(mode, lam, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    import random; random.seed(SEED)

    xi = loess_detrend(load_dat(DATA / "wti_2501.dat"))
    X, y = make_sequences(xi, CFG["lookback"])
    T_seq = len(X)

    # Precompute empirical kurt for moment mode (on full detrended return series)
    r_emp = np.diff(xi)
    kurt_emp = float(spstats.kurtosis(r_emp, fisher=False, bias=False))  # Pearson, unbiased
    print(f"[wti][{mode}] T={len(xi)}  T_seq={T_seq}  empirical kurt(r)={kurt_emp:.2f}",
          flush=True)

    # Chronological 85/15 split on sequence indices
    split = int(T_seq * TRAIN_FRAC)
    train_mask = np.ones(T_seq, dtype=bool)
    train_mask[split:] = False

    # Cushing exclusion: remove training sequences whose target date is in the window
    if mode == "cushing":
        m = CFG["lookback"]
        # sequence index i targets xi[i + m]
        target_idx = np.arange(T_seq) + m
        cushing_mask = (target_idx >= CUSHING_WINDOW[0]) & (target_idx < CUSHING_WINDOW[1])
        # additionally drop sequences whose context contains the window (leakage-free)
        ctx_hits = np.zeros(T_seq, dtype=bool)
        for i in range(T_seq):
            if (i + m > CUSHING_WINDOW[0]) and (i < CUSHING_WINDOW[1]):
                ctx_hits[i] = True
        drop_mask = cushing_mask | ctx_hits
        train_mask = train_mask & (~drop_mask)
        n_drop = int(drop_mask[:split].sum())
        print(f"  cushing: dropped {n_drop} training sequences whose target or context "
              f"intersects indices [{CUSHING_WINDOW[0]}, {CUSHING_WINDOW[1]})", flush=True)

    X_tr = X[train_mask]
    y_tr = y[train_mask]
    X_va = X[split:]
    y_va = y[split:]
    print(f"  train={len(X_tr)}  val={len(X_va)}", flush=True)

    Xt = torch.FloatTensor(X_tr).unsqueeze(-1).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_va).unsqueeze(-1).to(device)
    yv = torch.FloatTensor(y_va).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(CFG, arch=CFG["arch"]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    best, best_state, counter, best_ep = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pi, mu, sg = model(xb)
            nll = mdn_loss(pi, mu, sg, yb)
            if mode == "moment":
                pen, _ = moment_penalty_kurt(pi, mu, sg, kurt_emp)
                loss = nll + lam * pen
            else:
                loss = nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pi_v, mu_v, sg_v = model(Xv)
            nll_v = mdn_loss(pi_v, mu_v, sg_v, yv).item()
            if mode == "moment":
                pen_v, kurt_pred = moment_penalty_kurt(pi_v, mu_v, sg_v, kurt_emp)
                loss_v = nll_v + lam * pen_v.item()
                track = loss_v
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  ep {epoch+1:3d}  NLL={nll_v:+.4f}  pen={pen_v.item():.4f}  "
                          f"kurt_pred={kurt_pred.item():.2f}  total={loss_v:+.4f}",
                          flush=True)
            else:
                track = nll_v
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  ep {epoch+1:3d}  NLL={nll_v:+.4f}", flush=True)
        if track < best:
            best, counter, best_ep = track, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        if counter >= PATIENCE:
            print(f"  early stop ep {epoch+1}  best={best:+.4f} at ep {best_ep}",
                  flush=True)
            break
    model.load_state_dict(best_state)
    print(f"  training done in {time.time()-t0:.1f}s", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS / f"mdn_v2_wti_{mode}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {**CFG},
        "market": "wti",
        "mode": mode,
        "lambda_moment": lam if mode == "moment" else None,
        "phase1_best_val": float(best),
        "initial_history": xi[-CFG["lookback"]:].astype(np.float32),
        "lookback": CFG["lookback"],
    }, ckpt_path)
    print(f"  saved: {ckpt_path.name}", flush=True)
    return model, xi, best


# ---------------- Evaluation ----------------
def forward_all(model, xi, device):
    m = CFG["lookback"]
    T_eff = len(xi) - m
    X = np.stack([xi[t:t + m] for t in range(T_eff)], axis=0).astype(np.float32)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pis, mus, sigmas = [], [], []
        for s in range(0, T_eff, 512):
            p, mu, sg = model(X_t[s:s + 512])
            pis.append(p.cpu().numpy()); mus.append(mu.cpu().numpy()); sigmas.append(sg.cpu().numpy())
    return np.concatenate(pis), np.concatenate(mus), np.concatenate(sigmas)


def gmm_quantile(pi, mu, sigma, q, grid=2048):
    lo = float((mu - 8 * sigma).min()); hi = float((mu + 8 * sigma).max())
    xs = np.linspace(lo, hi, grid)
    cdf = np.zeros_like(xs)
    for k in range(len(pi)):
        cdf += pi[k] * norm.cdf(xs, mu[k], sigma[k])
    idx = int(np.clip(np.searchsorted(cdf, q), 1, grid - 1))
    x0, x1 = xs[idx - 1], xs[idx]; y0, y1 = cdf[idx - 1], cdf[idx]
    return float(x0) if y1 == y0 else float(x0 + (q - y0) / (y1 - y0) * (x1 - x0))


def gmm_cdf_at(pi, mu, sigma, x):
    return float(np.sum(pi * norm.cdf(x, mu, sigma)))


def kupiec(hits, alpha):
    n = len(hits); x = int(hits.sum()); phat = x / n if n > 0 else 0.0
    if phat <= 0 or phat >= 1: return 0.0, 1.0
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll1 = x * np.log(phat) + (n - x) * np.log(1 - phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen(hits):
    hits = hits.astype(int)
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0: return 0.0, 1.0
    p01 = n01 / n0; p11 = n11 / n1; p_any = (n01 + n11) / (n0 + n1)
    if any(x <= 0 or x >= 1 for x in [p_any, p01, p11]): return 0.0, 1.0
    ll0 = (n00 + n10) * np.log(1 - p_any) + (n01 + n11) * np.log(p_any)
    ll1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def run_var(PI, MU, SIGMA, xi):
    m = CFG["lookback"]; T_eff = len(xi) - m
    r_emp = xi[m:] - xi[m - 1:-1]
    out = {}
    for alpha in ALPHAS:
        var_l = np.empty(T_eff); var_r = np.empty(T_eff)
        for t in range(T_eff):
            q_lo = gmm_quantile(PI[t], MU[t], SIGMA[t], alpha)
            q_hi = gmm_quantile(PI[t], MU[t], SIGMA[t], 1 - alpha)
            xi_cur = float(xi[m - 1 + t])
            var_l[t] = q_lo - xi_cur; var_r[t] = q_hi - xi_cur
        hl = (r_emp < var_l).astype(int)
        hr = (r_emp > var_r).astype(int)
        for tail, h in [("left", hl), ("right", hr)]:
            n = len(h); x = int(h.sum())
            lr_uc, p_uc = kupiec(h, alpha)
            lr_ind, p_ind = christoffersen(h)
            lr_cc = lr_uc + lr_ind
            p_cc = float(1 - chi2.cdf(lr_cc, df=2))
            out[f"alpha={alpha:.2f}_{tail}"] = dict(
                n=n, exceed=x, rate=x / n, expected=alpha,
                lr_uc=lr_uc, p_uc=p_uc, lr_ind=lr_ind, p_ind=p_ind,
                lr_cc=lr_cc, p_cc=p_cc)
    return out


def ljung_box(x, lag=10):
    x = np.asarray(x) - np.mean(x); n = len(x); g0 = (x * x).mean(); q = 0.0
    for k in range(1, lag + 1):
        gk = (x[:-k] * x[k:]).mean()
        q += (gk / g0) ** 2 / (n - k)
    q *= n * (n + 2)
    return float(q), float(1 - chi2.cdf(q, df=lag))


def berkowitz(u):
    u = np.clip(u, 1e-6, 1 - 1e-6); z = norm.ppf(u); n = len(z)
    ll0 = np.sum(norm.logpdf(z, 0, 1))
    X = np.column_stack([np.ones(n - 1), z[:-1]]); y = z[1:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, rho = beta; resid = y - X @ beta; s2 = np.mean(resid ** 2)
    ll1 = np.sum(norm.logpdf(y, X @ beta, np.sqrt(s2)))
    ll1 += norm.logpdf(z[0], a / (1 - rho + 1e-9),
                       np.sqrt(s2 / max(1 - rho ** 2, 1e-6)))
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=3)), float(a), float(rho), float(np.sqrt(s2))


def run_pit(PI, MU, SIGMA, xi):
    m = CFG["lookback"]; T_eff = len(xi) - m
    target = xi[m:]
    u = np.array([gmm_cdf_at(PI[t], MU[t], SIGMA[t], float(target[t]))
                  for t in range(T_eff)])
    ks_s, ks_p = spstats.kstest(u, "uniform")
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    ad = spstats.anderson(z, dist="norm")
    ad_s = float(ad.statistic); lvls = [0.15, 0.10, 0.05, 0.025, 0.01]
    if ad_s < ad.critical_values[0]: ad_p = "> 0.15"
    elif ad_s >= ad.critical_values[-1]: ad_p = "< 0.01"
    else:
        ln_p = np.interp(ad_s, ad.critical_values, np.log(lvls))
        ad_p = f"~{np.exp(ln_p):.3f}"
    lb_s, lb_p = ljung_box(u - 0.5, 10)
    bk_s, bk_p, bk_mu, bk_rho, bk_sig = berkowitz(u)
    return dict(n=T_eff, u_mean=float(u.mean()), u_std=float(u.std(ddof=1)),
                ks=(ks_s, ks_p), ad=(ad_s, ad_p), lb=(lb_s, lb_p),
                bk=(bk_s, bk_p, bk_mu, bk_rho, bk_sig))


def sample_gmm(pi, mu, sigma, rng):
    k = rng.choice(len(pi), p=pi)
    return mu[k] + sigma[k] * rng.standard_normal()


def generate_paths(model, xi, device, n_paths=N_PATHS, seed=123):
    m = CFG["lookback"]; T = len(xi); rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, T), dtype=np.float32)
    paths[:, :m] = xi[:m]
    contexts = torch.FloatTensor(np.tile(xi[:m], (n_paths, 1))).unsqueeze(-1).to(device)
    t0 = time.time(); model.eval()
    with torch.no_grad():
        for t in range(m, T):
            pi, mu, sg = model(contexts)
            pi = pi.cpu().numpy(); mu = mu.cpu().numpy(); sg = sg.cpu().numpy()
            step = np.array([sample_gmm(pi[i], mu[i], sg[i], rng)
                             for i in range(n_paths)], dtype=np.float32)
            paths[:, t] = step
            new_ctx = np.column_stack([
                contexts.cpu().numpy().squeeze(-1)[:, 1:], step[:, None]
            ])
            contexts = torch.from_numpy(new_ctx).unsqueeze(-1).to(device)
            if (t - m + 1) % 500 == 0:
                print(f"  gen step {t-m+1}/{T-m}  elapsed={time.time()-t0:.1f}s",
                      flush=True)
    print(f"  MC gen done in {time.time()-t0:.1f}s", flush=True)
    return paths


def acf_single(x, maxlag):
    x = x - x.mean(); var = (x ** 2).mean()
    if var <= 0: return np.zeros(maxlag)
    return np.array([(x[:-k] * x[k:]).mean() / var for k in range(1, maxlag + 1)])


def acf_coverage(emp_series, sim_paths, maxlag=MAX_LAG):
    emp_acf = acf_single(emp_series, maxlag)
    sim_acfs = np.stack([acf_single(sim_paths[i], maxlag) for i in range(len(sim_paths))])
    lo = np.percentile(sim_acfs, 2.5, axis=0)
    hi = np.percentile(sim_acfs, 97.5, axis=0)
    in_band = (emp_acf >= lo) & (emp_acf <= hi)
    return float(np.mean(in_band))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("moment", "cushing"), required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1,
                    help="moment penalty weight (moment mode only)")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip MC generation + ACF coverage (faster for diagnostic)")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"device={device}  SEED={SEED}  mode={args.mode}  lambda={args.lam}",
          flush=True)
    print("=" * 70 + f"\n  WTI VARIANT -- {args.mode.upper()}\n" + "=" * 70, flush=True)

    model, xi, best = train(args.mode, args.lam, device)
    # Tag checkpoint filename with lambda for moment sweep
    if args.mode == "moment" and args.lam != 0.1:
        import shutil
        src = MODELS / "mdn_v2_wti_moment.pt"
        dst = MODELS / f"mdn_v2_wti_moment_lam{args.lam}.pt"
        shutil.copy(src, dst)
        print(f"  tagged checkpoint: {dst.name}", flush=True)
    PI, MU, SIGMA = forward_all(model, xi, device)

    print("\n--- VaR backtest ---", flush=True)
    var_res = run_var(PI, MU, SIGMA, xi)
    print("\n--- PIT diagnostics ---", flush=True)
    pit_res = run_pit(PI, MU, SIGMA, xi)

    cov_xi = cov_r2 = None
    if not args.skip_gen:
        print("\n--- MC paths + ACF coverage ---", flush=True)
        paths = generate_paths(model, xi, device, n_paths=N_PATHS, seed=123)
        cov_xi = acf_coverage(xi, paths, MAX_LAG)
        cov_r2 = acf_coverage(np.diff(xi) ** 2, np.diff(paths, axis=1) ** 2, MAX_LAG)

    # ---------- summary ----------
    lines = []
    lines.append("=" * 104)
    lines.append(f"  WTI variant -- {args.mode}  (baseline Gaussian MDN family)")
    lines.append(f"  Config: {CFG}")
    lines.append(f"  Phase 1 best val = {best:+.4f}   (ref Gauss baseline: -2.52)")
    if args.mode == "moment":
        lines.append(f"  lambda (moment weight) = {args.lam}   target kurt_emp = 31.9")
    lines.append("=" * 104)
    lines.append("\n[VaR backtest]")
    lines.append(f"  {'level':<10} {'tail':<6} {'N':>5} {'exc':>5} {'rate':>7} "
                 f"{'exp':>5} {'p_uc':>7} {'p_ind':>7} {'p_cc':>7}")
    for key, r in var_res.items():
        lvl, tail = key.split("_")
        lines.append(
            f"  {lvl:<10} {tail:<6} {r['n']:>5d} {r['exceed']:>5d} "
            f"{r['rate']:>7.4f} {r['expected']:>5.2f} "
            f"{r['p_uc']:>7.4f} {r['p_ind']:>7.4f} {r['p_cc']:>7.4f}"
        )
    lines.append("\n[PIT diagnostics]")
    ks_s, ks_p = pit_res["ks"]; ad_s, ad_p = pit_res["ad"]
    lb_s, lb_p = pit_res["lb"]; bk_s, bk_p, bk_mu, bk_rho, bk_sig = pit_res["bk"]
    lines.append(f"  N={pit_res['n']}  u_mean={pit_res['u_mean']:.4f}  u_std={pit_res['u_std']:.4f}")
    lines.append(f"  KS  stat={ks_s:.4f}  p={ks_p:.4f}")
    lines.append(f"  AD  stat={ad_s:.3f}  p={ad_p}")
    lines.append(f"  LB  stat={lb_s:.2f}   p={lb_p:.4f}")
    lines.append(f"  Berk LR={bk_s:.2f}  p={bk_p:.4f}   "
                 f"mu={bk_mu:+.3f}  rho={bk_rho:+.3f}  sigma={bk_sig:.3f}")
    if cov_xi is not None:
        lines.append(f"\n[ACF coverage]  xi={cov_xi:.0%}  r^2={cov_r2:.0%}")
    lines.append("\n[Reference: baseline Gaussian MDN]")
    lines.append("  VaR: 1/4 pass CC; left rates 0.44% (alpha=0.01), 2.74% (alpha=0.05)")
    lines.append("  PIT: u_mean=0.58, sigma_Berk=0.87")
    lines.append("  ACF: xi=58%, r^2=43%")
    lines.append("=" * 104)

    text = "\n".join(lines); print("\n" + text)
    suffix = f"_lam{args.lam}" if args.mode == "moment" and args.lam != 0.1 else ""
    out = DATA / f"wti_variant_{args.mode}{suffix}_summary.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved: {out.name}]")


if __name__ == "__main__":
    main()
