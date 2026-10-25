"""Train the Collusionyes circuit-breaker detector."""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DETECTOR_C, DETECTOR_FEATURES, RESULTS_DIR, SEED
from src.core_sim import extract_features, run_episode_arrays, run_switch_episode


def load_episodes() -> pd.DataFrame:
    path = RESULTS_DIR / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(
            "results/episodes.csv is missing. Run "
            "`python -m src.core_sim` first."
        )
    frame = pd.read_csv(path)
    missing = set(DETECTOR_FEATURES + ["label", "split"]).difference(
        frame.columns
    )
    if missing:
        raise ValueError(f"Episode table is missing columns: {sorted(missing)}")
    return frame


def threshold_rule(features: dict | pd.Series) -> tuple[int, float]:
    """Fixed-score fallback used by the dashboard demonstration."""

    score = float(features["collusion_score"])
    return int(score > 0.5), score


def train_and_evaluate(frame: pd.DataFrame | None = None) -> dict:
    """Train L1 logistic regression and export detector evidence."""

    frame = load_episodes() if frame is None else frame
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    x_train = train[DETECTOR_FEATURES]
    x_test = test[DETECTOR_FEATURES]
    y_train = train["label"].astype(int)
    y_test = test["label"].astype(int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=DETECTOR_C,
                    max_iter=5_000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore",
            message="Inconsistent values: penalty=l1 with l1_ratio=0.0.*",
        )
        model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    eligible = np.flatnonzero(tpr >= 0.90)
    fpr_at_90 = float(fpr[eligible[0]]) if eligible.size else 1.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    coefficients = model.named_steps["clf"].coef_[0]
    nonzero = [
        {"feature": feature, "coefficient": float(coefficient)}
        for feature, coefficient in sorted(
            zip(DETECTOR_FEATURES, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        if abs(coefficient) > 1e-10
    ]
    metrics = {
        "model": "StandardScaler + L1 logistic regression",
        "penalty": "l1",
        "solver": "liblinear",
        "C": DETECTOR_C,
        "split": {"train": len(x_train), "test": len(x_test), "stratified": True},
        "training_behavior_noise": float(
            train["behavior_noise"].dropna().unique().max()
        ),
        "test_behavior_noise": float(
            test["behavior_noise"].fillna(0.0).abs().max()
        ),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "fpr_at_tpr90": fpr_at_90,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "nonzero_coefficients": nonzero,
    }
    (RESULTS_DIR / "detector_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="ascii",
    )

    thresholds = np.linspace(0.0, 1.0, 101)
    curve_rows = []
    for threshold in thresholds:
        predictions_at_threshold = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_test, predictions_at_threshold, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        curve_rows.append(
            {
                "threshold": float(threshold),
                "fpr": fp / max(fp + tn, 1),
                "tpr": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
            }
        )
    pd.DataFrame(curve_rows).to_csv(
        RESULTS_DIR / "halt_tradeoff.csv",
        index=False,
    )
    joblib.dump(model, RESULTS_DIR / "detector_model.joblib")
    return metrics


def rolling_probabilities(
    regime: str = "mixed",
    seed: int = SEED,
    window: int = 50,
    step: int = 10,
) -> pd.DataFrame:
    """Score a single episode over rolling windows for near-real-time use."""

    model_path = RESULTS_DIR / "detector_model.joblib"
    if not model_path.exists():
        train_and_evaluate()
    model = joblib.load(model_path)
    if regime == "mixed":
        raw = run_switch_episode(seed=seed, switch_round=500)
    else:
        raw = run_episode_arrays(regime=regime, seed=seed, behavior_noise=0.0)
    prices = raw["prices"]
    demand = raw["demand"]
    rows = []
    for end in range(window, len(prices) + 1, step):
        features = extract_features(
            prices[end - window : end],
            demand[end - window : end],
        )
        probability = float(
            model.predict_proba(
                pd.DataFrame([features])[DETECTOR_FEATURES]
            )[0, 1]
        )
        rows.append(
            {
                "regime": raw["regime"],
                "seed": int(seed),
                "window": int(window),
                "start_round": int(end - window + 1),
                "end_round": int(end),
                "collusion_probability": probability,
                "halt": int(probability >= 0.5),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        RESULTS_DIR / "rolling_detector_timeseries.csv",
        index=False,
    )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(
        result["end_round"],
        result["collusion_probability"],
        color="#C2410C",
        linewidth=2,
    )
    ax.axhline(
        0.5,
        color="#334155",
        linestyle="--",
        linewidth=1,
        label="halt threshold",
    )
    ax.axvline(500, color="#0F766E", linestyle=":", linewidth=1.5)
    ax.annotate(
        "collusion begins",
        xy=(500, 0.9),
        xytext=(8, 0),
        textcoords="offset points",
        color="#0F766E",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Collusion probability")
    ax.set_ylim(0, 1)
    ax.set_title("Rolling 50-round circuit-breaker signal")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "figures" / "rolling_detector_probability.png",
        dpi=180,
    )
    plt.close(fig)
    return result


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result, indent=2))
    assert result["auc"] >= 0.85
    assert result["nonzero_coefficients"]
    rolling = rolling_probabilities()
    assert len(rolling) > 0
    assert rolling["collusion_probability"].between(0, 1).all()
    assert (
        rolling["collusion_probability"].max()
        - rolling["collusion_probability"].min()
        > 0.1
    )
    print("Detector sanity check passed")
