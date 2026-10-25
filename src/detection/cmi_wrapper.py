"""Adapter from continuous Collusim prices to the teammate's calibrated CMI screen."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "src" / "external"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from infocollusion.detect import screen  # noqa: E402


DEFAULT_STATE_BINS = (1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75)


def discretize_price_actions(
    prices: np.ndarray,
    hold_threshold: float = 0.01,
) -> np.ndarray:
    """Map own-price changes to Down=0, Hold=1, Up=2."""

    values = np.asarray(prices, dtype=float)
    if values.ndim != 2:
        raise ValueError("prices must have shape (rounds, agents)")
    if len(values) < 2:
        raise ValueError("at least two rounds are required")
    changes = np.diff(values, axis=0)
    actions = np.ones_like(changes, dtype=np.int64)
    actions[changes < -hold_threshold] = 0
    actions[changes > hold_threshold] = 2
    return actions


def public_price_states(
    prices: np.ndarray,
    bins: tuple[float, ...] = DEFAULT_STATE_BINS,
) -> np.ndarray:
    """Summarize the previous public price level, excluding the current action."""

    values = np.asarray(prices, dtype=float)
    previous_mean = values[:-1].mean(axis=1)
    return np.digitize(previous_mean, bins).astype(np.int64)


def _action_entropy(actions: np.ndarray) -> float:
    counts = np.bincount(actions.ravel(), minlength=3).astype(float)
    probabilities = counts / max(counts.sum(), 1.0)
    return float(
        -sum(
            probability * math.log2(probability)
            for probability in probabilities
            if probability > 0
        )
    )


def screen_prices(
    prices: np.ndarray,
    *,
    hold_threshold: float = 0.01,
    n_perm: int = 200,
    level: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the calibrated screen on a continuous ``(rounds, agents)`` price log."""

    values = np.asarray(prices, dtype=float)
    actions = discretize_price_actions(values, hold_threshold=hold_threshold)
    states = public_price_states(values)
    rows = screen(
        actions,
        states,
        n_perm=n_perm,
        level=level,
        seed=seed,
    )
    if not rows:
        raise ValueError("CMI screen returned no agent pairs")
    top = dict(rows[0])
    pair = tuple(int(index) for index in top["pair"])
    return {
        "top_pair": f"{pair[0]}-{pair[1]}",
        "cmi_bits": float(top["cmi_bits"]),
        "net_influence_bits": float(top["net_influence_bits"]),
        "z": float(top["z"]),
        "p_value": float(top["p"]),
        "flagged": bool(top["flagged"]),
        "action_entropy_bits": _action_entropy(actions),
        "frozen": bool((actions == actions.flat[0]).all()),
    }


def screen_long_trajectories(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("condition", "seed"),
    entity_column: str = "seller_id",
    round_column: str = "round",
    price_column: str = "price",
    hold_threshold: float = 0.01,
) -> pd.DataFrame:
    """Screen each multi-agent trajectory stored in long form."""

    required = {
        *group_columns,
        entity_column,
        round_column,
        price_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Trajectory table missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for group_key, part in frame.groupby(list(group_columns), sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        matrix = (
            part.pivot(
                index=round_column,
                columns=entity_column,
                values=price_column,
            )
            .sort_index()
            .to_numpy(dtype=float)
        )
        result = screen_prices(matrix, hold_threshold=hold_threshold)
        rows.append(
            {
                **dict(zip(group_columns, group_key)),
                **result,
                "agents": int(matrix.shape[1]),
                "rounds": int(matrix.shape[0]),
            }
        )
    return pd.DataFrame(rows)
