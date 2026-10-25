"""Tests for the calibrated CMI adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.cmi_wrapper import (
    discretize_price_actions,
    screen_long_trajectories,
    screen_prices,
)


def test_price_moves_bin_to_down_hold_up() -> None:
    prices = np.array(
        [
            [1.00, 1.00],
            [0.90, 1.00],
            [0.90, 1.10],
            [1.10, 1.10],
        ]
    )
    actions = discretize_price_actions(prices)
    assert actions.tolist() == [
        [0, 1],
        [1, 2],
        [2, 1],
    ]


def test_frozen_prices_have_zero_cmi_and_are_not_flagged() -> None:
    prices = np.full((50, 2), 2.5)
    result = screen_prices(prices, n_perm=100, seed=7)
    assert result["frozen"] is True
    assert result["cmi_bits"] == 0.0
    assert result["flagged"] is False


def test_screen_long_trajectories_returns_top_pair() -> None:
    rows = []
    for seed in (1, 2):
        for round_number in range(1, 51):
            pair_move = 2.0 + 0.2 * (round_number % 2)
            rows.extend(
                [
                    {
                        "condition": "baseline",
                        "seed": seed,
                        "round": round_number,
                        "seller_id": "a",
                        "price": pair_move,
                    },
                    {
                        "condition": "baseline",
                        "seed": seed,
                        "round": round_number,
                        "seller_id": "b",
                        "price": pair_move,
                    },
                    {
                        "condition": "baseline",
                        "seed": seed,
                        "round": round_number,
                        "seller_id": "c",
                        "price": 1.0 + 0.1 * ((round_number * seed) % 3),
                    },
                ]
            )
    result = screen_long_trajectories(pd.DataFrame(rows))
    assert len(result) == 2
    assert set(result["top_pair"]).issubset({"0-1", "0-2", "1-2"})
    assert result["cmi_bits"].ge(0).all()
