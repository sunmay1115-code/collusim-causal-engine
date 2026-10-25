"""One-call calibrated collusion screen for any discrete action log (the entry point for teammates).

    from infocollusion.detect import screen
    rows = screen(actions, states)   # actions: (T, n_agents) ints/labels; states: (T,), (T, k) or None

Each row is one agent pair, sorted most suspicious first:
    pair, cmi_bits (Miller-Madow corrected I(a_i; a_j | s)), net_influence_bits (Bonjour Gamma, min of
    both directions), z and p (within-state permutation test), flagged (p < 0.05 / #pairs and both net
    influences > 0, i.e. the calibrated rule of the report).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from infocollusion.estimators import encode, influence_matrix, net_influence, permutation_test

LN2 = np.log(2)


def screen(actions, states=None, n_perm: int = 500, level: float = 0.05, seed: int = 0) -> list[dict]:
    a = np.column_stack([encode(col) for col in np.asarray(actions, dtype=object).T])
    if states is not None:
        st = np.asarray(states, dtype=object)
        states = encode(map(tuple, st) if st.ndim == 2 else st)   # multi-column state -> one code
    s = states
    n = a.shape[1]
    if n < 2:
        raise ValueError("need at least two agents")
    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(n), 2))
    n_perm = max(n_perm, int(np.ceil(2 * len(pairs) / level)))   # else Bonferroni p can never be reached
    g = influence_matrix(a, s, miller_madow=True)
    big = net_influence(g) if n >= 3 else g
    rows = []
    for i, j in pairs:
        t = permutation_test(a[:, i], a[:, j], s, n_perm=n_perm, rng=rng)
        rows.append(dict(pair=(i, j), cmi_bits=float(t.gamma / LN2),
                         net_influence_bits=float(min(big[i, j], big[j, i])) / LN2, z=float(t.z), p=float(t.p),
                         flagged=bool(t.p < level / len(pairs) and big[i, j] > 0 and big[j, i] > 0)))
    return sorted(rows, key=lambda r: -r["z"])
