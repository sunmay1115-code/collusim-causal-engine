"""Validation tests for the Collusim v2 observation-to-causality bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import RESULTS_DIR
from src.collusim.environment import run_episode
from src.llm_track.estimate import _validation_frame, calibrate_parameters


def test_calibration_recovers_known_beta() -> None:
    params = calibrate_parameters(_validation_frame())
    assert abs(params["beta"] - 0.7) <= 0.05


def test_blind_agents_price_lower_than_perfect_information() -> None:
    perfect = run_episode(seed=7, visibility_loss=0.0)
    blind = run_episode(seed=7, visibility_loss=1.0)
    assert blind["mean_markup"] < perfect["mean_markup"]


def test_history_reset_injects_one_near_nash_round() -> None:
    control = run_episode(seed=7)
    reset = run_episode(seed=7, history_reset=True, reset_round=50)
    reset_round = int(reset["history_reset_round"])
    assert reset["history_reset_applied"] is True
    assert control["history_reset_applied"] is False
    assert reset["history_reset_price"] == pytest.approx(1.473)
    assert reset["prices"][reset_round - 1] == pytest.approx(1.473)
    assert not np.allclose(
        reset["prices"][reset_round - 1],
        control["prices"][reset_round - 1],
    )


def test_collusim_artifacts() -> None:
    required = [
        "collusim_params.json",
        "collusim_frontier.csv",
        "collusim_rct_raw.csv",
    ]
    for name in required:
        assert (RESULTS_DIR / name).exists(), name

    frontier = pd.read_csv(RESULTS_DIR / "collusim_frontier.csv")
    strongest = frontier.sort_values("strength").groupby("lever").tail(1)
    assert set(strongest["lever"]) == {
        "precision",
        "delay",
        "visibility",
        "history_reset",
        "hard_ban",
    }
    established = strongest.loc[strongest["lever"] != "history_reset"]
    assert (established["collusion_reduction"] > 0).all()
    assert (frontier["efficiency_ci_low"] <= frontier["efficiency_ratio"]).all()
    assert (frontier["efficiency_ratio"] <= frontier["efficiency_ci_high"]).all()
    reset = frontier.loc[frontier["lever"] == "history_reset"]
    assert len(reset) == 1


def test_detector_artifacts() -> None:
    episodes = pd.read_csv(RESULTS_DIR / "episodes.csv")
    assert len(episodes) == 600
    assert episodes.groupby("regime").size().to_dict() == {
        "colluding": 200,
        "competitive": 200,
        "noisy": 200,
    }
    metrics = pd.read_json(RESULTS_DIR / "detector_metrics.json", typ="series")
    assert metrics["auc"] >= 0.85
    assert metrics["fpr_at_tpr90"] <= 0.30
    assert metrics["training_behavior_noise"] == 0.1
    assert metrics["test_behavior_noise"] == 0.0
    tradeoff = pd.read_csv(RESULTS_DIR / "halt_tradeoff.csv")
    assert len(tradeoff) == 101
    rolling = pd.read_csv(RESULTS_DIR / "rolling_detector_timeseries.csv")
    assert rolling["collusion_probability"].between(0, 1).all()
    assert (
        rolling["collusion_probability"].max()
        - rolling["collusion_probability"].min()
        > 0.1
    )

