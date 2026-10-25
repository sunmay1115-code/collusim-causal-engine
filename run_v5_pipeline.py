"""Rebuild Collusim, the CMI detector, and V5 objective evidence."""

from __future__ import annotations

from src.collusim.experiments import run_rct
from src.collusim.objectives import write_objective_outputs
from src.core_sim import build_episode_dataset
from src.detector import train_and_evaluate


def main() -> int:
    episodes = build_episode_dataset()
    detector = train_and_evaluate(episodes)
    print(
        f"Detector AUC={detector['auc']:.3f}, "
        f"FPR@90%TPR={detector['fpr_at_tpr90']:.3f}"
    )

    frontier = run_rct()
    summary = write_objective_outputs()
    winner = summary["objective_1_5"]["winner"]
    print(frontier.to_string(index=False))
    print(
        "Objective 1.1 positive control: "
        f"{summary['objective_1_1']['positive_control_pass']}"
    )
    if winner is None:
        print("Objective 1.5: no restriction passed all evidence gates.")
    else:
        print(
            "Objective 1.5 winner: "
            f"{winner['lever']} at {winner['strength']:.2f}; "
            f"markup reduction={winner['markup_reduction_pct']:.1%}, "
            f"CMI reduction={winner['cmi_reduction_pct']:.1%}"
        )
    print("V5 pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
