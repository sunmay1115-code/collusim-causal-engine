"""Tests for V5 objective evidence tables."""

from __future__ import annotations

import json

import pandas as pd

from config import RESULTS_DIR


def test_positive_controls_are_explicit_and_passing() -> None:
    controls = pd.read_csv(
        RESULTS_DIR / "objective_1_1_positive_control.csv"
    )
    assert set(controls["control"]) == {
        "rct_hard_ban",
        "scripted_benchmark",
    }
    assert controls["pass"].all()
    assert (controls["markup_gap_ci_low"] > 0.05).all()
    assert (controls["cmi_gap_ci_low"] > 0.02).all()


def test_ablations_include_price_detector_and_state() -> None:
    ablations = pd.read_csv(
        RESULTS_DIR / "objective_1_1_variable_ablations.csv"
    )
    assert set(ablations["lever"]) == {
        "precision",
        "delay",
        "visibility",
        "history_reset",
        "hard_ban",
    }
    required = {
        "mean_markup",
        "cmi_bits",
        "markup_reduction_pct",
        "cmi_reduction_pct",
        "detector_interpretation",
        "conditioning_state",
        "action_definition",
    }
    assert required.issubset(ablations.columns)
    assert ablations["conditioning_state"].notna().all()


def test_restriction_winner_is_valid_signal_restriction() -> None:
    ranking = pd.read_csv(
        RESULTS_DIR / "objective_1_5_restriction_ranking.csv"
    )
    eligible = ranking.loc[ranking["detector_valid_restriction"]]
    assert not eligible.empty
    winner = eligible.sort_values("efficiency_ratio", ascending=False).iloc[0]
    assert winner["lever"] in {
        "precision",
        "delay",
        "visibility",
        "history_reset",
    }
    assert winner["markup_reduction_pct"] >= 0.10
    assert winner["cmi_reduction_pct"] >= 0.10
    assert winner["restriction_status"] != "Freeze (Failed Restriction)"

    summary = json.loads(
        (RESULTS_DIR / "v5_objective_summary.json").read_text(
            encoding="ascii"
        )
    )
    assert summary["objective_1_1"]["positive_control_pass"] is True
    assert summary["objective_1_5"]["winner"]["lever"] == winner["lever"]
    assert (RESULTS_DIR / "restriction_ranking.csv").exists()


def test_history_reset_runs_through_strict_gates() -> None:
    ranking = pd.read_csv(
        RESULTS_DIR / "objective_1_5_restriction_ranking.csv"
    )
    reset = ranking.loc[ranking["lever"] == "history_reset"].iloc[0]
    assert reset["positive_control_pass"]
    assert pd.notna(reset["markup_reduction_ci_low"])
    assert pd.notna(reset["cmi_reduction_ci_low"])
    assert 0.0 <= reset["cmi_frozen_share"] < 0.50
    assert reset["detector_valid_restriction"] in (False, True)
