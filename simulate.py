"""Simulation of MCUSUM-CoDa and T2-CoDa out-of-control signals in ilr coordinates.

All randomness flows through numpy Generator objects created from explicit seeds.
"""
import itertools
import numpy as np
from scipy.stats import chi2

# In-control parameters (ilr coordinates, pivot basis). The values are chosen to
# resemble a bronze alloy (p = 3) and a lithium-ion cell (p = 4) composition;
# they are simulation settings, not estimates from plant data.
CASES = {
    "bronze": dict(
        parts=["Cu", "Sn", "Zn"],
        mu0=np.array([2.225, 1.440]),
        Sigma=np.array([[1.019, -0.015], [-0.015, 1.099]]),
    ),
    "lithium": dict(
        parts=["cathode", "anode", "electrolyte", "separator"],
        mu0=np.array([0.971, 0.486, 0.230]),
        Sigma=np.array([[0.104, 0.006, 0.016], [0.006, 0.083, -0.010], [0.016, -0.010, 0.108]]),
    ),
}
DELTAS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
K_REF = 0.5
ARL0 = 370.0
BURN_IN = 20          # in-control observations before the shift occurs
MAX_STEPS = 20000


def patterns(d):
    """All non-empty subsets of the d coordinates, ordered by size then lexicographically."""
    pats = []
    for r in range(1, d + 1):
        for comb in itertools.combinations(range(d), r):
            v = np.zeros(d, int)
            v[list(comb)] = 1
            pats.append(tuple(v))
    return pats


def mcusum_step(u, x, Sinv, k):
    """One Crosier (1988) MCUSUM update. u, x: (n, d) arrays of deviations from mu0."""
    v = u + x
    C = np.sqrt(np.einsum("ij,jk,ik->i", v, Sinv, v))
    fac = np.where(C > k, 1.0 - k / np.maximum(C, 1e-12), 0.0)
    u_new = v * fac[:, None]
    D = np.sqrt(np.einsum("ij,jk,ik->i", u_new, Sinv, u_new))
    return u_new, D, C <= k


def arl0_mcusum(d, h, k=K_REF, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros((n, d)); rl = np.zeros(n); alive = np.ones(n, bool); t = 0
    I = np.eye(d)
    while alive.any() and t < MAX_STEPS:
        t += 1
        idx = np.where(alive)[0]
        u[idx], D, _ = mcusum_step(u[idx], rng.standard_normal((len(idx), d)), I, k)
        sig = D > h
        rl[idx[sig]] = t; alive[idx[sig]] = False
    return rl.mean(), rl.std(ddof=1) / np.sqrt(n)


def calibrate_h(d, target=ARL0, k=K_REF, lo=3.0, hi=12.0, n=20000, iters=14, seed=11):
    """Bisection for the MCUSUM control limit giving the target in-control ARL (standardized data)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a, _ = arl0_mcusum(d, mid, k, n, seed)
        lo, hi = (mid, hi) if a < target else (lo, mid)
    h = 0.5 * (lo + hi)
    a, se = arl0_mcusum(d, h, k, 2 * n, seed + 1)
    return h, a, se


def simulate_signals(case, chart, h, delta, pattern, n_signals, rng, k=K_REF):
    """Generate n_signals out-of-control signals for one shift pattern and size.

    Each run starts from u_0 = 0, receives BURN_IN in-control observations (runs that
    signal during the burn-in are false alarms and are discarded), and then a sustained
    shift of delta * sd_j in every coordinate j with pattern[j] = 1 until the chart signals.
    Returns features at the signal time and the post-shift run length.
    """
    mu0, S = CASES[case]["mu0"], CASES[case]["Sigma"]
    d = len(mu0); L = np.linalg.cholesky(S); Sinv = np.linalg.inv(S); sd = np.sqrt(np.diag(S))
    shift = delta * sd * np.array(pattern)
    ucl_t2 = chi2.ppf(1 - 1 / ARL0, d)
    out = {"obs": [], "u": [], "runmean": [], "rl": []}
    got = 0
    while got < n_signals:
        n = int((n_signals - got) * 1.3) + 50
        u = np.zeros((n, d)); s_sum = np.zeros((n, d)); s_cnt = np.zeros(n)
        alive = np.ones(n, bool); fa = np.zeros(n, bool)
        for t in range(1, BURN_IN + MAX_STEPS + 1):
            idx = np.where(alive)[0]
            if len(idx) == 0:
                break
            x = rng.standard_normal((len(idx), d)) @ L.T
            if t > BURN_IN:
                x = x + shift
            if chart == "mcusum":
                u[idx], D, reset = mcusum_step(u[idx], x, Sinv, k)
                # running mean of deviations since the last reset of the cumulative vector
                s_sum[idx] = np.where(reset[:, None], 0.0, s_sum[idx] + x)
                s_cnt[idx] = np.where(reset, 0.0, s_cnt[idx] + 1)
                sig = D > h
            else:  # Hotelling T2
                T2 = np.einsum("ij,jk,ik->i", x, Sinv, x)
                sig = T2 > ucl_t2
            if t <= BURN_IN:
                fa[idx[sig]] = True; alive[idx[sig]] = False
                continue
            for j in np.where(sig)[0]:
                i = idx[j]
                out["obs"].append(x[j] / sd)
                out["u"].append(u[i] / sd)
                cnt = max(s_cnt[i], 1.0)
                out["runmean"].append(s_sum[i] / cnt / sd)
                out["rl"].append(t - BURN_IN)
            alive[idx[sig]] = False
        got = len(out["rl"])
    return {kk: np.array(v[:n_signals]) for kk, v in out.items()}


def build_dataset(case, chart, h, n_per_cell, seed):
    """Signals for every (delta, pattern) cell. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    d = len(CASES[case]["mu0"]); pats = patterns(d)
    X = {"obs": [], "u": [], "runmean": []}; y = []; dl = []; rl = []
    for delta in DELTAS:
        for ci, p in enumerate(pats):
            s = simulate_signals(case, chart, h, delta, p, n_per_cell, rng)
            for kk in X:
                X[kk].append(s[kk])
            y.append(np.full(n_per_cell, ci)); dl.append(np.full(n_per_cell, delta)); rl.append(s["rl"])
    res = {kk: np.vstack(v) for kk, v in X.items()}
    res.update(y=np.concatenate(y), delta=np.concatenate(dl), rl=np.concatenate(rl))
    return res
