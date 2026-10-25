"""Information-theoretic dependence estimators for collusion detection.

Implements the individual influence (gamma) and net influence (Gamma) of Bonjour, Aggarwal & Bhargava,
"Information-Theoretic Approach to Detect Collusion in Multi-Agent Games", UAI 2022 (no public code
existed; this is a from-scratch implementation of their Eqs. 3-6), plus the extensions used in this
project:

* Miller-Madow bias correction of the plug-in conditional mutual information;
* a within-state permutation null, giving a z-score / p-value for each pair instead of a hand-set
  threshold alpha;
* a run-level "excess dependence" score for coalitions larger than a pair, where net influence is blind
  by construction (it subtracts the strongest other partner).

All quantities are in nats. Actions and states are arrays of non-negative ints (use `encode` first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def encode(values) -> np.ndarray:
    """Map arbitrary hashable labels to 0..K-1 ints (stable order of first appearance)."""
    mapping: dict = {}
    return np.array([mapping.setdefault(v, len(mapping)) for v in values], dtype=np.int64)


def _cmi_from_codes(x: np.ndarray, y: np.ndarray, s: np.ndarray, kx: int, ky: int, ks: int,
                    miller_madow: bool) -> float:
    """sum_s p(s) I(X;Y | S=s), plug-in estimate, optionally Miller-Madow corrected."""
    n = x.size
    joint = np.bincount((s * kx + x) * ky + y, minlength=ks * kx * ky).reshape(ks, kx, ky).astype(float)
    ns = joint.sum(axis=(1, 2))
    px = joint.sum(axis=2)  # counts N(s, x)
    py = joint.sum(axis=1)  # counts N(s, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        # p(x,y|s) log [ p(x,y|s) / (p(x|s) p(y|s)) ] = N(sxy)/N(s) * log[ N(sxy) N(s) / (N(sx) N(sy)) ]
        ratio = joint * ns[:, None, None] / (px[:, :, None] * py[:, None, :])
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    cmi = terms.sum() / n
    if miller_madow:
        # MI bias for independent X,Y with Kx, Ky occupied cells is ~(Kx-1)(Ky-1)/(2 N_s), per state.
        occ_x = (px > 0).sum(axis=1)
        occ_y = (py > 0).sum(axis=1)
        valid = ns > 0
        bias = ((occ_x - 1) * (occ_y - 1))[valid] / (2.0 * n)  # p(s) * 1/(2 N_s) = 1/(2n)
        cmi -= bias.sum()
    return float(cmi)


def conditional_mi(x, y, s=None, miller_madow: bool = False) -> float:
    """I(X;Y|S) for discrete samples. s=None means a single (trivial) state, i.e. plain MI."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    s = np.zeros_like(x) if s is None else np.asarray(s, dtype=np.int64)
    return _cmi_from_codes(x, y, s, int(x.max()) + 1, int(y.max()) + 1, int(s.max()) + 1, miller_madow)


def _within_state_permutation(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index permutation that shuffles samples only among rows sharing the same state."""
    order = np.lexsort((rng.random(s.size), s))      # rows grouped by state, random order inside groups
    sorted_idx = np.argsort(s, kind="stable")         # rows grouped by state, original order inside
    perm = np.empty_like(order)
    perm[sorted_idx] = order
    return perm


@dataclass
class PairTest:
    gamma: float          # observed individual influence (bias-corrected if requested)
    null_mean: float
    null_sd: float
    z: float
    p: float


def permutation_test(x, y, s=None, n_perm: int = 200, miller_madow: bool = True,
                     rng: np.random.Generator | None = None) -> PairTest:
    """Test I(X;Y|S)=0 by shuffling Y within each state (preserves p(x|s) and p(y|s))."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    s = np.zeros_like(x) if s is None else np.asarray(s, dtype=np.int64)
    kx, ky, ks = int(x.max()) + 1, int(y.max()) + 1, int(s.max()) + 1
    obs = _cmi_from_codes(x, y, s, kx, ky, ks, miller_madow)
    null = np.array([
        _cmi_from_codes(x, y[_within_state_permutation(s, rng)], s, kx, ky, ks, miller_madow)
        for _ in range(n_perm)
    ])
    sd = float(null.std(ddof=1)) if n_perm > 1 else 0.0
    z = (obs - null.mean()) / sd if sd > 0 else 0.0
    p = (1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)
    return PairTest(obs, float(null.mean()), sd, float(z), float(p))


def g_test(x, y, s=None) -> tuple[float, float]:
    """Asymptotic test of I(X;Y|S)=0: G = 2 N I_plugin ~ chi2 with sum_s (Kx_s-1)(Ky_s-1) dof.
    Returns (G, p). Fast alternative to the permutation test for large samples."""
    from scipy.stats import chi2
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    s = np.zeros_like(x) if s is None else np.asarray(s, dtype=np.int64)
    kx, ky, ks = int(x.max()) + 1, int(y.max()) + 1, int(s.max()) + 1
    g = 2 * x.size * _cmi_from_codes(x, y, s, kx, ky, ks, False)
    joint = np.bincount((s * kx + x) * ky + y, minlength=ks * kx * ky).reshape(ks, kx, ky)
    dof = int((((joint.sum(2) > 0).sum(1) - 1).clip(0) * ((joint.sum(1) > 0).sum(1) - 1).clip(0)).sum())
    return g, (float(chi2.sf(g, dof)) if dof > 0 else 1.0)


def influence_matrix(actions: np.ndarray, states=None, miller_madow: bool = False) -> np.ndarray:
    """gamma[i, j] = I(a_i; a_j | s) for every pair (simultaneous, fully observable game, Eq. 6).

    actions: (T, n) int array, one column per agent. states: (T,) shared state codes or None.
    Symmetric with a zero diagonal.
    """
    actions = np.asarray(actions, dtype=np.int64)
    n = actions.shape[1]
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g[i, j] = g[j, i] = conditional_mi(actions[:, i], actions[:, j], states, miller_madow)
    return g


def net_influence(gamma: np.ndarray) -> np.ndarray:
    """Gamma[i, j] = gamma[i, j] - max_{k != i, j} gamma[k, j]  (Bonjour et al. Eq. 4). Needs n >= 3."""
    n = gamma.shape[0]
    if n < 3:
        raise ValueError("net influence needs at least 3 agents")
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                others = [gamma[k, j] for k in range(n) if k not in (i, j)]
                out[i, j] = gamma[i, j] - max(others)
    return out


def bonjour_detect(gamma: np.ndarray, alpha: float = 0.05):
    """Paper's decision rule: collusion iff exactly one pair has Gamma(i;j) and Gamma(j;i) >= alpha.

    gamma is in nats (as returned here); alpha is in bits, which is the unit under which the paper's
    alpha=0.05 is consistent with its reported CP=0.2 / CP=0.1 detection rates."""
    big = net_influence(gamma) / np.log(2)
    pairs = [(i, j) for i in range(len(big)) for j in range(i + 1, len(big))
             if big[i, j] >= alpha and big[j, i] >= alpha]
    return pairs[0] if len(pairs) == 1 else None


def pair_scores(gamma: np.ndarray) -> dict[tuple[int, int], float]:
    """Symmetric pair score min(Gamma(i;j), Gamma(j;i)) used to rank candidate pairs."""
    big = net_influence(gamma)
    n = len(big)
    return {(i, j): float(min(big[i, j], big[j, i])) for i in range(n) for j in range(i + 1, n)}
