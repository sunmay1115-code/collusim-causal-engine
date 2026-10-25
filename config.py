"""Central configuration for the LLM behavioural statistics track."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT / "data" / "raw"
APP_VERSION = "v5"

SEED = 42
TEST_SIZE = 0.30
EPSILON = 0.05
N_STATES_TARGET = 500
MIN_ROWS_PER_MODEL = 300
CI_LEVEL = 0.90
N_BOOTSTRAP = 1000

MODELS = ("llama3:8b", "qwen2.5:7b")

FEATURES = [
    "competitor_price",
    "own_price",
    "demand",
    "round_norm",
    "lagged_demand",
    "market_mean_price",
]
DETECTOR_FEATURES = [
    "mean_markup",
    "max_markup",
    "markup_persistence",
    "price_variance",
    "cross_agent_corr",
    "sync_index",
    "mutual_information",
    "price_dispersion",
    "stability",
    "autocorrelation_lag1",
    "volatility",
    "undercut_rate",
    "match_rate",
    "strategy_entropy",
    "lead_lag_corr",
]
STRATEGIES = ("undercut", "match", "overcut")

# A benchmark prior representing an even mix of competitive and matching
# scripted behavior. It is documented and used only for the H5 bridge test.
SCRIPTED_STRATEGY_DISTRIBUTION = {
    "undercut": 0.40,
    "match": 0.40,
    "overcut": 0.20,
}

COLLUSIM_MARKET = {
    "n_agents": 2,
    "n_rounds": 100,
    "c": 1.0,
    "M": 100.0,
    "lam": 2.0,
    "p_min": 1.0,
    "p_max": 3.0,
    "initial_price": 3.0,
    "blind_competitive_pull": 0.08,
    "consumer_value": 3.0,
    "human_markup": 0.25,
    "human_liquidity": 0.60,
    "human_price_noise": 0.03,
    "history_reset_price": 1.473,
}

COLLUSIM_GRIDS = {
    "precision": [0.0, 0.05, 0.1, 0.2, 0.4],
    "delay": [0, 1, 3, 5, 10],
    "visibility_loss": [0.0, 0.2, 0.5, 0.8, 1.0],
    "history_reset": [False, True],
}

COLLUSIM_SEEDS = 20
DETECTOR_C = 0.014

