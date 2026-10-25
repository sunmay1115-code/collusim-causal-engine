"""Calibrate Collusim parameters from the LLM behavior dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import MODELS, RESULTS_DIR, SEED


PARAMS_PATH = RESULTS_DIR / "collusim_params.json"
RESIDUALS_PATH = RESULTS_DIR / "collusim_residuals.npz"


def _calibration_design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    renamed = frame.rename(
        columns={
            "own_price": "own_last_price",
            "round_norm": "round_number",
        }
    )
    x = sm.add_constant(
        renamed[
            ["competitor_price", "own_last_price", "round_number"]
        ].to_numpy(dtype=float),
        has_constant="add",
    )
    y = frame["llm_price"].to_numpy(dtype=float)
    return x, y


def calibrate_parameters(frame: pd.DataFrame) -> dict:
    """Extract non-causal OLS parameters for the proxy policy."""

    if frame.empty:
        raise ValueError("Cannot calibrate on an empty dataset")
    x, y = _calibration_design(frame)
    result = sm.OLS(y, x).fit()
    residual_variance = float(np.var(result.resid, ddof=x.shape[1]))
    return {
        "intercept": float(result.params[0]),
        "beta": float(result.params[1]),
        "own_last_coef": float(result.params[2]),
        "round_coef": float(result.params[3]),
        "sigma2": residual_variance,
        "sigma": float(np.sqrt(max(residual_variance, 0.0))),
        "n": int(len(frame)),
        "r2": float(result.rsquared),
    }


def estimate_collusim_params(
    frame: pd.DataFrame | None = None,
) -> dict:
    """Fit pooled and per-model calibration parameters and export JSON."""

    if frame is None:
        frame = pd.read_csv(RESULTS_DIR / "llm_behavior.csv")
    frame = frame.loc[frame["is_valid"]].copy()
    pooled = calibrate_parameters(frame)
    x, y = _calibration_design(frame)
    pooled_residuals = y - x @ np.asarray(
        [
            pooled["intercept"],
            pooled["beta"],
            pooled["own_last_coef"],
            pooled["round_coef"],
        ]
    )
    residuals = {"pooled": pooled_residuals}
    by_model = {
        model: calibrate_parameters(model_frame)
        for model, model_frame in frame.groupby("model_name")
    }
    for model, model_frame in frame.groupby("model_name"):
        x_model, y_model = _calibration_design(model_frame)
        params = by_model[model]
        residuals[model] = y_model - x_model @ np.asarray(
            [
                params["intercept"],
                params["beta"],
                params["own_last_coef"],
                params["round_coef"],
            ]
        )
    np.savez_compressed(RESIDUALS_PATH, **residuals)
    payload = {
        "dataset_run_id": str(frame["run_id"].iloc[0]),
        "data_source": str(frame["data_source"].iloc[0]),
        "formula": (
            "llm_price ~ competitor_price + own_last_price + round_number"
        ),
        "causal": False,
        "residual_file": "results/collusim_residuals.npz",
        "residual_key": "pooled",
        "default": pooled,
        "by_model": by_model,
        "model_order": list(MODELS),
    }
    PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return payload


def _validation_frame(seed: int = 71, n: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    competitor = rng.uniform(1.0, 3.0, size=n)
    own = rng.uniform(1.0, 3.0, size=n)
    round_norm = rng.uniform(size=n)
    price = (
        0.25
        + 0.70 * competitor
        + 0.08 * own
        + 0.03 * round_norm
        + rng.normal(0.0, 0.18, size=n)
    )
    return pd.DataFrame(
        {
            "llm_price": price,
            "competitor_price": competitor,
            "own_last_price": own,
            "round_number": round_norm,
        }
    )


if __name__ == "__main__":
    validation = _validation_frame()
    recovered = calibrate_parameters(validation)
    print(
        f"Recovered beta={recovered['beta']:.4f}, "
        f"sigma={recovered['sigma']:.4f}"
    )
    assert abs(recovered["beta"] - 0.7) <= 0.05
    print("Calibration sanity check passed")
