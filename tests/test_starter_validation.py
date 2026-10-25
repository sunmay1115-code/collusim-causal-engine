"""Tests for the offline official-starter proof-of-concept artifacts."""

from __future__ import annotations

import pandas as pd

from config import RESULTS_DIR


def test_q11_visible_history_artifacts() -> None:
    baseline = pd.read_csv(RESULTS_DIR / "starter_q11_baseline.csv")
    ablation = pd.read_csv(RESULTS_DIR / "starter_q11_ablation.csv")
    required = {"round", "visible_rounds", "mean_price", "source"}
    assert required.issubset(baseline.columns)
    assert required.issubset(ablation.columns)
    assert set(baseline["visible_rounds"]) == {100}
    assert set(ablation["visible_rounds"]) == {0}
    assert len(baseline) == 30
    assert len(ablation) == 30
    assert baseline["mean_price"].mean() > ablation["mean_price"].mean()
    assert baseline["mean_price"].mean() > baseline[
        "competitive_benchmark"
    ].mean()
    assert ablation["mean_price"].mean() < ablation[
        "competitive_benchmark"
    ].mean()
    assert baseline["source"].str.contains("local_mock").all()


def test_q15_message_cap_artifacts() -> None:
    baseline = pd.read_csv(RESULTS_DIR / "starter_q15_baseline.csv")
    ablation = pd.read_csv(RESULTS_DIR / "starter_q15_ablation.csv")
    expected = {
        "average_price",
        "seller_profit",
        "trade_volume",
        "buyer_welfare",
    }
    assert set(baseline.columns) == expected
    assert set(ablation.columns) == expected
    assert len(baseline) == 1
    assert len(ablation) == 1
    baseline = baseline.iloc[0]
    ablation = ablation.iloc[0]
    assert baseline["trade_volume"] > ablation["trade_volume"]
    assert baseline["buyer_welfare"] > ablation["buyer_welfare"]
