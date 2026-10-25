"""Non-information-theoretic pairwise baselines and shared pair-ranking utilities."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .estimators import influence_matrix, net_influence, permutation_test


def pearson_abs(x, y) -> float:
    """|Pearson r| between integer action codes: the 'ordinary action correlation' baseline."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def agreement_z(x, y) -> float:
    """|z| of the same-action rate against its expectation under independent marginals."""
    x, y = np.asarray(x), np.asarray(y)
    n = x.size
    k = int(max(x.max(), y.max())) + 1
    px = np.bincount(x, minlength=k) / n
    py = np.bincount(y, minlength=k) / n
    e = float(px @ py)
    obs = float(np.mean(x == y))
    sd = np.sqrt(max(e * (1 - e), 1e-12) / n)
    return abs(obs - e) / sd


def score_pairs(actions: np.ndarray, states: np.ndarray | None, n_perm: int = 0,
                rng: np.random.Generator | None = None) -> dict[str, dict[tuple[int, int], float]]:
    """Every pair score used in the experiments, keyed by method then by (i, j)."""
    actions = np.asarray(actions)
    n = actions.shape[1]
    pairs = list(combinations(range(n), 2))
    out: dict[str, dict] = {
        "pearson": {p: pearson_abs(actions[:, p[0]], actions[:, p[1]]) for p in pairs},
        "agreement": {p: agreement_z(actions[:, p[0]], actions[:, p[1]]) for p in pairs},
    }
    g_unc = influence_matrix(actions, None)
    out["mi_uncond"] = {p: g_unc[p] for p in pairs}
    if n >= 3:
        big_unc = net_influence(g_unc)
        out["netinf_uncond"] = {p: min(big_unc[p], big_unc[p[::-1]]) for p in pairs}
    if states is not None:
        g = influence_matrix(actions, states)
        out["cmi"] = {p: g[p] for p in pairs}
        if n >= 3:
            big = net_influence(g)
            out["netinf"] = {p: min(big[p], big[p[::-1]]) for p in pairs}
    if n_perm:
        rng = rng or np.random.default_rng(0)
        tests = {p: permutation_test(actions[:, p[0]], actions[:, p[1]], states, n_perm=n_perm, rng=rng)
                 for p in pairs}
        out["perm_z"] = {p: t.z for p, t in tests.items()}
        out["perm_p"] = {p: -np.log10(t.p) for p, t in tests.items()}   # larger = more suspicious
    return out


def top_pair(scores: dict[tuple[int, int], float]) -> tuple[int, int]:
    return max(scores, key=scores.get)


def auroc(pos, neg) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())
