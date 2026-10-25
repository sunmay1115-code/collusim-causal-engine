"""Randomized signal-degradation experiments for Collusim."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import COLLUSIM_GRIDS, COLLUSIM_SEEDS, RESULTS_DIR, SEED
from deferred.llm_track.common import bootstrap_ci
from src.collusim.environment import run_episode
from src.detection.cmi_wrapper import screen_prices


LEVER_COLUMNS = {
    "precision": "precision",
    "delay": "delay",
    "visibility": "visibility_loss",
    "history_reset": "history_reset",
}


def _cmi_for_result(result: dict, seed: int) -> dict:
    return screen_prices(
        result["prices"],
        hold_threshold=0.01,
        n_perm=100,
        seed=seed,
    )


def _classify_restriction(
    baseline: dict,
    condition: dict,
) -> tuple[str, float, float]:
    markup_drop = (
        baseline["markup"] - condition["markup"]
    ) / max(abs(baseline["markup"]), 1e-9)
    cmi_drop = (
        baseline["cmi"] - condition["cmi"]
    ) / max(abs(baseline["cmi"]), 1e-9)
    if cmi_drop >= 0.10 and markup_drop < 0.10:
        label = "Freeze (Failed Restriction)"
    elif cmi_drop >= 0.10 and markup_drop >= 0.10:
        label = "Broken (Successful Restriction)"
    elif markup_drop >= 0.10:
        label = "Price Reduced, CMI Signal Static"
    else:
        label = "No Material Effect"
    return label, float(markup_drop), float(cmi_drop)


def _run_condition(
    lever: str,
    strength: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for index in range(COLLUSIM_SEEDS):
        seed = SEED + index
        argument = {LEVER_COLUMNS[lever]: strength}
        if lever == "history_reset":
            argument["history_reset"] = bool(strength)
        result = run_episode(seed=seed, **argument)
        cmi = _cmi_for_result(result, seed)
        rows.append(
            {
                "lever": lever,
                "strength": float(strength),
                "seed": seed,
                "mean_price": float(result["prices"].mean()),
                "mean_markup": result["mean_markup"],
                "total_welfare": result["total_welfare"],
                "blind_share": result["blind_share"],
                "cmi_bits": cmi["cmi_bits"],
                "cmi_z": cmi["z"],
                "cmi_p_value": cmi["p_value"],
                "cmi_flagged": int(cmi["flagged"]),
                "cmi_top_pair": cmi["top_pair"],
                "cmi_action_entropy": cmi["action_entropy_bits"],
                "cmi_frozen": int(cmi["frozen"]),
                "history_reset_applied": int(
                    result["history_reset_applied"]
                ),
                "history_reset_round": int(result["history_reset_round"]),
                "history_reset_price": float(result["history_reset_price"]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "markup": float(frame["mean_markup"].mean()),
        "price": float(frame["mean_price"].mean()),
        "welfare": float(frame["total_welfare"].mean()),
        "cmi": float(frame["cmi_bits"].mean()),
        "cmi_z": float(frame["cmi_z"].mean()),
        "cmi_flagged_share": float(frame["cmi_flagged"].mean()),
        "cmi_action_entropy": float(frame["cmi_action_entropy"].mean()),
        "cmi_frozen_share": float(frame["cmi_frozen"].mean()),
    }


def _run_hard_ban() -> tuple[pd.DataFrame, dict]:
    rows = []
    for index in range(COLLUSIM_SEEDS):
        seed = SEED + index
        result = run_episode(seed=seed, agent_type="human")
        cmi = _cmi_for_result(result, seed)
        rows.append(
            {
                "lever": "hard_ban",
                "strength": 1.0,
                "seed": seed,
                "mean_price": float(result["prices"].mean()),
                "mean_markup": result["mean_markup"],
                "total_welfare": result["total_welfare"],
                "blind_share": result["blind_share"],
                "cmi_bits": cmi["cmi_bits"],
                "cmi_z": cmi["z"],
                "cmi_p_value": cmi["p_value"],
                "cmi_flagged": int(cmi["flagged"]),
                "cmi_top_pair": cmi["top_pair"],
                "cmi_action_entropy": cmi["action_entropy_bits"],
                "cmi_frozen": int(cmi["frozen"]),
                "history_reset_applied": int(
                    result["history_reset_applied"]
                ),
                "history_reset_round": int(result["history_reset_round"]),
                "history_reset_price": float(result["history_reset_price"]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "markup": float(frame["mean_markup"].mean()),
        "price": float(frame["mean_price"].mean()),
        "welfare": float(frame["total_welfare"].mean()),
        "cmi": float(frame["cmi_bits"].mean()),
        "cmi_z": float(frame["cmi_z"].mean()),
        "cmi_flagged_share": float(frame["cmi_flagged"].mean()),
        "cmi_action_entropy": float(frame["cmi_action_entropy"].mean()),
        "cmi_frozen_share": float(frame["cmi_frozen"].mean()),
    }


def run_rct() -> pd.DataFrame:
    """Run paired seed experiments for all configured restriction levers."""

    baseline_rows, baseline = _run_condition("precision", 0.0)
    baseline_rows["lever"] = "baseline"
    raw_frames = [baseline_rows]
    frontier = []
    for lever_key, strengths in COLLUSIM_GRIDS.items():
        lever = "visibility" if lever_key == "visibility_loss" else lever_key
        for strength in strengths:
            if float(strength) == 0.0:
                continue
            condition_rows, aggregate = _run_condition(lever, float(strength))
            raw_frames.append(condition_rows)
            reduction = baseline["markup"] - aggregate["markup"]
            welfare_cost = baseline["welfare"] - aggregate["welfare"]
            cmi_reduction = baseline["cmi"] - aggregate["cmi"]
            label, markup_drop_pct, cmi_drop_pct = _classify_restriction(
                baseline,
                aggregate,
            )
            ratio = reduction / max(abs(welfare_cost), 1e-9)
            paired = baseline_rows.merge(
                condition_rows,
                on="seed",
                suffixes=("_baseline", "_condition"),
            )
            per_seed = (
                paired["mean_markup_baseline"]
                - paired["mean_markup_condition"]
            ) / np.maximum(
                np.abs(
                    paired["total_welfare_baseline"]
                    - paired["total_welfare_condition"]
                ),
                1e-9,
            )
            _, ci_low, ci_high = bootstrap_ci(
                per_seed.to_numpy(dtype=float),
                n_boot=1000,
                ci=0.90,
                seed=SEED,
            )
            frontier.append(
                {
                    "lever": lever,
                    "strength": float(strength),
                    "collusion_reduction": reduction,
                    "mean_price": aggregate["price"],
                    "mean_markup": aggregate["markup"],
                    "cmi_bits": aggregate["cmi"],
                    "cmi_z": aggregate["cmi_z"],
                    "cmi_reduction": cmi_reduction,
                    "cmi_flagged_share": aggregate["cmi_flagged_share"],
                    "cmi_action_entropy": aggregate["cmi_action_entropy"],
                    "cmi_frozen_share": aggregate["cmi_frozen_share"],
                    "markup_reduction_pct": markup_drop_pct,
                    "cmi_reduction_pct": cmi_drop_pct,
                    "restriction_status": label,
                    "welfare_cost": welfare_cost,
                    "efficiency_ratio": ratio,
                    "efficiency_ci_low": min(ci_low, ratio),
                    "efficiency_ci_high": max(ci_high, ratio),
                }
            )

    ban_rows, ban_aggregate = _run_hard_ban()
    raw_frames.append(ban_rows)
    reduction = baseline["markup"] - ban_aggregate["markup"]
    welfare_cost = baseline["welfare"] - ban_aggregate["welfare"]
    cmi_reduction = baseline["cmi"] - ban_aggregate["cmi"]
    label, markup_drop_pct, cmi_drop_pct = _classify_restriction(
        baseline,
        ban_aggregate,
    )
    paired = baseline_rows.merge(
        ban_rows,
        on="seed",
        suffixes=("_baseline", "_condition"),
    )
    per_seed = (
        paired["mean_markup_baseline"] - paired["mean_markup_condition"]
    ) / np.maximum(
        np.abs(
            paired["total_welfare_baseline"]
            - paired["total_welfare_condition"]
        ),
        1e-9,
    )
    _, ci_low, ci_high = bootstrap_ci(per_seed.to_numpy(dtype=float))
    ratio = reduction / max(abs(welfare_cost), 1e-9)
    frontier.append(
        {
            "lever": "hard_ban",
            "strength": 1.0,
            "collusion_reduction": reduction,
            "mean_price": ban_aggregate["price"],
            "mean_markup": ban_aggregate["markup"],
            "cmi_bits": ban_aggregate["cmi"],
            "cmi_z": ban_aggregate["cmi_z"],
            "cmi_reduction": cmi_reduction,
            "cmi_flagged_share": ban_aggregate["cmi_flagged_share"],
            "cmi_action_entropy": ban_aggregate["cmi_action_entropy"],
            "cmi_frozen_share": ban_aggregate["cmi_frozen_share"],
            "markup_reduction_pct": markup_drop_pct,
            "cmi_reduction_pct": cmi_drop_pct,
            "restriction_status": label,
            "welfare_cost": welfare_cost,
            "efficiency_ratio": ratio,
            "efficiency_ci_low": min(ci_low, ratio),
            "efficiency_ci_high": max(ci_high, ratio),
        }
    )

    pd.concat(raw_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "collusim_rct_raw.csv",
        index=False,
    )
    result = pd.DataFrame(frontier).sort_values(["lever", "strength"])
    result.to_csv(RESULTS_DIR / "collusim_frontier.csv", index=False)
    return result


if __name__ == "__main__":
    frontier = run_rct()
    print(frontier.to_string(index=False))
    strongest = frontier.sort_values("strength").groupby("lever").tail(1)
    assert (strongest["collusion_reduction"] > 0).all()
    best = frontier.sort_values("efficiency_ratio", ascending=False).iloc[0]
    ban = frontier.loc[frontier["lever"] == "hard_ban"].iloc[0]
    non_ban_best = frontier.loc[
        frontier["lever"] != "hard_ban"
    ].sort_values("efficiency_ratio", ascending=False).iloc[0]
    assert non_ban_best["efficiency_ratio"] > ban["efficiency_ratio"]
    print(
        f"Best lever: {best['lever']} at {best['strength']:.2f}; "
        f"reduction={best['collusion_reduction']:.3f}, "
        f"welfare_cost={best['welfare_cost']:.3f}"
    )
    print(
        f"Hard ban efficiency={ban['efficiency_ratio']:.3f}; "
        f"best non-ban={non_ban_best['efficiency_ratio']:.3f}"
    )
    print("Collusim RCT sanity check passed")
