"""Evidence tables for team-brief objectives 1.1 and 1.5."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from config import CI_LEVEL, N_BOOTSTRAP, RESULTS_DIR, SEED
from deferred.llm_track.common import bootstrap_ci


DETECTOR_SPEC = {
    "method": "Miller-Madow CMI with within-state permutation test",
    "multiple_testing": "Bonferroni over agent pairs",
    "actions": (
        "own-price change: down < -0.01, hold within +/-0.01, up > 0.01"
    ),
    "conditioning_state": (
        "previous-round mean public price, binned at "
        "1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75"
    ),
    "test_level": 0.05,
}

MIN_MARKUP_GAP = 0.05
MIN_CMI_GAP = 0.02
MIN_MARKUP_REDUCTION = 0.10
MIN_CMI_REDUCTION = 0.10


def _ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    return bootstrap_ci(
        np.asarray(values, dtype=float),
        n_boot=N_BOOTSTRAP,
        ci=CI_LEVEL,
        seed=seed,
    )


def _independent_gap_ci(
    treated: np.ndarray,
    control: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        treated_sample = rng.choice(treated, size=treated.size, replace=True)
        control_sample = rng.choice(control, size=control.size, replace=True)
        estimates[index] = treated_sample.mean() - control_sample.mean()
    tail = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    point = float(treated.mean() - control.mean())
    return point, float(low), float(high)


def _required_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(
            "V5 objective analysis requires regenerated CMI results; "
            f"missing columns: {sorted(missing)}"
        )


def _rct_positive_control(raw: pd.DataFrame) -> dict[str, Any]:
    required = {
        "lever",
        "seed",
        "mean_markup",
        "cmi_bits",
        "cmi_flagged",
        "cmi_action_entropy",
    }
    _required_columns(raw, required)
    baseline = raw.loc[raw["lever"] == "baseline"].copy()
    comparator = raw.loc[raw["lever"] == "hard_ban"].copy()
    paired = baseline.merge(
        comparator,
        on="seed",
        suffixes=("_baseline", "_control"),
    )
    if paired.empty:
        raise ValueError("No paired baseline and hard-ban control rows.")

    markup_gap = (
        paired["mean_markup_baseline"] - paired["mean_markup_control"]
    ).to_numpy()
    cmi_gap = (
        paired["cmi_bits_baseline"] - paired["cmi_bits_control"]
    ).to_numpy()
    flag_gap = (
        paired["cmi_flagged_baseline"] - paired["cmi_flagged_control"]
    ).to_numpy()
    markup_point, markup_low, markup_high = _ci(markup_gap, SEED + 401)
    cmi_point, cmi_low, cmi_high = _ci(cmi_gap, SEED + 402)
    flag_point, flag_low, flag_high = _ci(flag_gap, SEED + 403)
    passed = bool(markup_low > MIN_MARKUP_GAP and cmi_low > MIN_CMI_GAP)
    detail = {
        "control": "rct_hard_ban",
        "treated": "calibrated_llm",
        "untreated": "hard_ban_human",
        "n": int(len(paired)),
        "treated_mean_markup": float(paired["mean_markup_baseline"].mean()),
        "untreated_mean_markup": float(paired["mean_markup_control"].mean()),
        "markup_gap": markup_point,
        "markup_gap_ci_low": markup_low,
        "markup_gap_ci_high": markup_high,
        "treated_mean_cmi_bits": float(paired["cmi_bits_baseline"].mean()),
        "untreated_mean_cmi_bits": float(paired["cmi_bits_control"].mean()),
        "cmi_gap_bits": cmi_point,
        "cmi_gap_ci_low": cmi_low,
        "cmi_gap_ci_high": cmi_high,
        "flag_gap": flag_point,
        "flag_gap_ci_low": flag_low,
        "flag_gap_ci_high": flag_high,
        "required_markup_gap": MIN_MARKUP_GAP,
        "required_cmi_gap": MIN_CMI_GAP,
        "treated_flag_share": float(
            paired["cmi_flagged_baseline"].mean()
        ),
        "untreated_flag_share": float(
            paired["cmi_flagged_control"].mean()
        ),
        "treated_mean_z": float(paired["cmi_z_baseline"].mean())
        if "cmi_z_baseline" in paired
        else float("nan"),
        "untreated_mean_z": float(paired["cmi_z_control"].mean())
        if "cmi_z_control" in paired
        else float("nan"),
        "pass": passed,
    }
    return detail


def _scripted_positive_control(
    episodes: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "regime",
        "mean_markup",
        "cmi_bits",
        "cmi_flagged",
        "cmi_action_entropy",
    }
    _required_columns(episodes, required)
    colluding = episodes.loc[episodes["regime"] == "colluding"]
    competitive = episodes.loc[episodes["regime"] == "competitive"]
    if colluding.empty or competitive.empty:
        raise ValueError("Scripted positive control needs both regimes.")
    markup_gap, markup_low, markup_high = _independent_gap_ci(
        colluding["mean_markup"].to_numpy(),
        competitive["mean_markup"].to_numpy(),
        SEED + 411,
    )
    cmi_gap, cmi_low, cmi_high = _independent_gap_ci(
        colluding["cmi_bits"].to_numpy(),
        competitive["cmi_bits"].to_numpy(),
        SEED + 412,
    )
    passed = bool(markup_low > MIN_MARKUP_GAP and cmi_low > MIN_CMI_GAP)
    detail = {
        "control": "scripted_benchmark",
        "treated": "colluding",
        "untreated": "competitive",
        "n_treated": int(len(colluding)),
        "n_untreated": int(len(competitive)),
        "treated_mean_markup": float(colluding["mean_markup"].mean()),
        "untreated_mean_markup": float(competitive["mean_markup"].mean()),
        "markup_gap": markup_gap,
        "markup_gap_ci_low": markup_low,
        "markup_gap_ci_high": markup_high,
        "treated_mean_cmi_bits": float(colluding["cmi_bits"].mean()),
        "untreated_mean_cmi_bits": float(competitive["cmi_bits"].mean()),
        "cmi_gap_bits": cmi_gap,
        "cmi_gap_ci_low": cmi_low,
        "cmi_gap_ci_high": cmi_high,
        "treated_flag_share": float(colluding["cmi_flagged"].mean()),
        "untreated_flag_share": float(competitive["cmi_flagged"].mean()),
        "treated_mean_action_entropy": float(
            colluding["cmi_action_entropy"].mean()
        ),
        "untreated_mean_action_entropy": float(
            competitive["cmi_action_entropy"].mean()
        ),
        "required_markup_gap": MIN_MARKUP_GAP,
        "required_cmi_gap": MIN_CMI_GAP,
        "pass": passed,
    }
    row = {
        "control": detail["control"],
        "treated": detail["treated"],
        "untreated": detail["untreated"],
        "n_treated": detail["n_treated"],
        "n_untreated": detail["n_untreated"],
        "treated_mean_markup": detail["treated_mean_markup"],
        "untreated_mean_markup": detail["untreated_mean_markup"],
        "markup_gap": detail["markup_gap"],
        "markup_gap_ci_low": detail["markup_gap_ci_low"],
        "markup_gap_ci_high": detail["markup_gap_ci_high"],
        "treated_mean_cmi_bits": detail["treated_mean_cmi_bits"],
        "untreated_mean_cmi_bits": detail["untreated_mean_cmi_bits"],
        "cmi_gap_bits": detail["cmi_gap_bits"],
        "cmi_gap_ci_low": detail["cmi_gap_ci_low"],
        "cmi_gap_ci_high": detail["cmi_gap_ci_high"],
        "required_markup_gap": detail["required_markup_gap"],
        "required_cmi_gap": detail["required_cmi_gap"],
        "pass": detail["pass"],
    }
    return detail, row


def _interpret_detector(
    price_reduced: bool,
    detector_reduced: bool,
    frozen_share: float,
) -> str:
    if frozen_share >= 0.50:
        return "Frozen prices; CMI is uninformative"
    if price_reduced and detector_reduced:
        return "Price and dependence both reduced"
    if detector_reduced:
        return "Detector falls without price relief"
    if price_reduced:
        return "Price relief; dependence persists"
    return "No material restriction"


def build_objective_tables(
    frontier: pd.DataFrame | None = None,
    raw: pd.DataFrame | None = None,
    episodes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build 1.1 ablation and 1.5 restriction evidence from saved outputs."""

    frontier = (
        pd.read_csv(RESULTS_DIR / "collusim_frontier.csv")
        if frontier is None
        else frontier.copy()
    )
    raw = (
        pd.read_csv(RESULTS_DIR / "collusim_rct_raw.csv")
        if raw is None
        else raw.copy()
    )
    episodes = (
        pd.read_csv(RESULTS_DIR / "episodes.csv")
        if episodes is None
        else episodes.copy()
    )
    _required_columns(
        frontier,
        {
            "lever",
            "strength",
            "mean_markup",
            "cmi_bits",
            "cmi_flagged_share",
            "restriction_status",
            "welfare_cost",
            "efficiency_ratio",
        },
    )
    _required_columns(raw, {"lever", "strength", "seed", "mean_markup", "cmi_bits"})

    rct_control = _rct_positive_control(raw)
    scripted_control, scripted_row = _scripted_positive_control(episodes)
    positive_control_pass = bool(
        rct_control["pass"] and scripted_control["pass"]
    )
    control_rows = [
        {
            "control": rct_control["control"],
            "treated": rct_control["treated"],
            "untreated": rct_control["untreated"],
            "n_treated": rct_control["n"],
            "n_untreated": rct_control["n"],
            "treated_mean_markup": rct_control["treated_mean_markup"],
            "untreated_mean_markup": rct_control["untreated_mean_markup"],
            "markup_gap": rct_control["markup_gap"],
            "markup_gap_ci_low": rct_control["markup_gap_ci_low"],
            "markup_gap_ci_high": rct_control["markup_gap_ci_high"],
            "treated_mean_cmi_bits": rct_control["treated_mean_cmi_bits"],
            "untreated_mean_cmi_bits": rct_control[
                "untreated_mean_cmi_bits"
            ],
            "cmi_gap_bits": rct_control["cmi_gap_bits"],
            "cmi_gap_ci_low": rct_control["cmi_gap_ci_low"],
            "cmi_gap_ci_high": rct_control["cmi_gap_ci_high"],
            "required_markup_gap": MIN_MARKUP_GAP,
            "required_cmi_gap": MIN_CMI_GAP,
            "treated_flag_share": rct_control["treated_flag_share"],
            "untreated_flag_share": rct_control["untreated_flag_share"],
            "treated_mean_z": rct_control["treated_mean_z"],
            "untreated_mean_z": rct_control["untreated_mean_z"],
            "pass": rct_control["pass"],
        },
        scripted_row,
    ]
    controls = pd.DataFrame(control_rows)

    baseline_rows = raw.loc[raw["lever"] == "baseline"].copy()
    baseline_markup = float(baseline_rows["mean_markup"].mean())
    baseline_cmi = float(baseline_rows["cmi_bits"].mean())
    baseline_welfare = float(baseline_rows["total_welfare"].mean())

    rows = []
    for index, row in frontier.reset_index(drop=True).iterrows():
        key = (
            (raw["lever"] == "baseline")
            if row["lever"] == "hard_ban"
            else (
                (raw["lever"] == row["lever"])
                & np.isclose(
                    raw["strength"].astype(float),
                    float(row["strength"]),
                )
            )
        )
        condition_rows = raw.loc[key].copy()
        if row["lever"] == "hard_ban":
            condition_rows = raw.loc[raw["lever"] == "hard_ban"].copy()
        paired = baseline_rows.merge(
            condition_rows,
            on="seed",
            suffixes=("_baseline", "_condition"),
        )
        if paired.empty:
            raise ValueError(f"No paired seeds for {row['lever']}.")
        markup_gap = (
            paired["mean_markup_baseline"]
            - paired["mean_markup_condition"]
        ).to_numpy()
        cmi_gap_values = (
            paired["cmi_bits_baseline"] - paired["cmi_bits_condition"]
        ).to_numpy()
        markup_point, markup_low, markup_high = _ci(
            markup_gap,
            SEED + 500 + index,
        )
        cmi_point, cmi_low, cmi_high = _ci(
            cmi_gap_values,
            SEED + 700 + index,
        )
        markup_reduction_pct = markup_point / max(abs(baseline_markup), 1e-9)
        cmi_reduction_pct = cmi_point / max(abs(baseline_cmi), 1e-9)
        frozen_values = (
            condition_rows["cmi_frozen"]
            if "cmi_frozen" in condition_rows
            else pd.Series(0, index=condition_rows.index)
        )
        entropy_values = (
            condition_rows["cmi_action_entropy"]
            if "cmi_action_entropy" in condition_rows
            else pd.Series(np.nan, index=condition_rows.index)
        )
        frozen_share = float(frozen_values.mean())
        price_reduced = bool(
            markup_reduction_pct >= MIN_MARKUP_REDUCTION
            and markup_low > 0.0
        )
        detector_reduced = bool(
            cmi_reduction_pct >= MIN_CMI_REDUCTION
            and cmi_point >= MIN_CMI_GAP
            and cmi_low > 0.0
            and frozen_share < 0.50
        )
        is_signal_restriction = row["lever"] != "hard_ban"
        eligible = bool(
            positive_control_pass
            and is_signal_restriction
            and price_reduced
            and detector_reduced
        )
        rows.append(
            {
                "lever": row["lever"],
                "strength": float(row["strength"]),
                "mean_price": float(row.get("mean_price", np.nan)),
                "mean_markup": float(row["mean_markup"]),
                "markup_reduction": markup_point,
                "markup_reduction_pct": markup_reduction_pct,
                "markup_reduction_ci_low": markup_low,
                "markup_reduction_ci_high": markup_high,
                "cmi_bits": float(row["cmi_bits"]),
                "cmi_z": float(row.get("cmi_z", np.nan)),
                "cmi_reduction": cmi_point,
                "cmi_reduction_pct": cmi_reduction_pct,
                "cmi_reduction_ci_low": cmi_low,
                "cmi_reduction_ci_high": cmi_high,
                "cmi_flagged_share": float(row["cmi_flagged_share"]),
                "cmi_frozen_share": frozen_share,
                "mean_action_entropy_bits": float(entropy_values.mean()),
                "welfare_cost": float(row["welfare_cost"]),
                "welfare_change_pct": float(row["welfare_cost"])
                / max(abs(baseline_welfare), 1e-9),
                "efficiency_ratio": float(row["efficiency_ratio"]),
                "efficiency_ci_low": float(row.get("efficiency_ci_low", np.nan)),
                "efficiency_ci_high": float(
                    row.get("efficiency_ci_high", np.nan)
                ),
                "restriction_status": str(row["restriction_status"]),
                "detector_interpretation": _interpret_detector(
                    price_reduced,
                    detector_reduced,
                    frozen_share,
                ),
                "positive_control_pass": positive_control_pass,
                "is_signal_restriction": is_signal_restriction,
                "detector_valid_restriction": eligible,
                "conditioning_state": DETECTOR_SPEC["conditioning_state"],
                "action_definition": DETECTOR_SPEC["actions"],
            }
        )

    ablation = pd.DataFrame(rows)
    ranking = ablation.loc[
        ablation["is_signal_restriction"]
    ].sort_values(
        ["detector_valid_restriction", "efficiency_ratio"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    eligible = ranking.loc[ranking["detector_valid_restriction"]]
    if eligible.empty:
        winner = None
    else:
        winner = eligible.iloc[0].to_dict()
    strongest = (
        ablation.loc[ablation["is_signal_restriction"]]
        .sort_values("strength")
        .groupby("lever", as_index=False)
        .tail(1)
    )
    contributing_variables = strongest.loc[
        strongest["detector_valid_restriction"], "lever"
    ].tolist()

    summary = {
        "version": "v5",
        "detector_spec": DETECTOR_SPEC,
        "objective_1_1": {
            "question": "Which variables are necessary for collusion to emerge?",
            "positive_control_pass": positive_control_pass,
            "rct_positive_control": rct_control,
            "scripted_positive_control": scripted_control,
            "baseline_mean_markup": baseline_markup,
            "baseline_cmi_bits": baseline_cmi,
            "variables_whose_strongest_ablation_passed": contributing_variables,
            "interpretation": (
                "These are simulation variables whose strongest removal reduced "
                "both the price level and state-conditional dependence while "
                "the positive controls passed. This is causal evidence inside "
                "Collusim, not proof about every LLM market."
            ),
        },
        "objective_1_5": {
            "question": (
                "Which single restriction removes the most collusion at the "
                "least cost?"
            ),
            "selection_rule": (
                "Among positive-control-passing signal restrictions, require a "
                "positive lower confidence bound for both markup and CMI "
                "reductions, then maximize welfare efficiency ratio. A CMI "
                "fall caused by frozen prices is ineligible."
            ),
            "winner": winner,
            "hard_ban": ablation.loc[
                ablation["lever"] == "hard_ban"
            ].iloc[0].to_dict(),
        },
    }
    return controls, ablation, ranking, summary


def write_objective_outputs() -> dict[str, Any]:
    """Write the V5 objective tables and machine-readable summary."""

    controls, ablation, ranking, summary = build_objective_tables()
    controls.to_csv(
        RESULTS_DIR / "objective_1_1_positive_control.csv",
        index=False,
    )
    ablation.to_csv(
        RESULTS_DIR / "objective_1_1_variable_ablations.csv",
        index=False,
    )
    ranking.to_csv(
        RESULTS_DIR / "objective_1_5_restriction_ranking.csv",
        index=False,
    )
    ranking.to_csv(
        RESULTS_DIR / "restriction_ranking.csv",
        index=False,
    )
    (RESULTS_DIR / "v5_objective_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="ascii",
    )
    return summary


if __name__ == "__main__":
    result = write_objective_outputs()
    print(json.dumps(result, indent=2, allow_nan=False))
