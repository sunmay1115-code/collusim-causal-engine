"""Generate the labeled scripted-market dataset used by the detector."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SEED
from src.detection.cmi_wrapper import screen_prices


MARKET = {
    "n_agents": 2,
    "n_rounds": 1000,
    "c": 1.0,
    "M": 100.0,
    "lam": 2.0,
    "p_min": 1.0,
    "p_max": 3.0,
    "price_step": 0.01,
    "collusive_price": 2.0,
    "punishment_rounds": 50,
}

FEATURE_COLS = [
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

REGIMES = ("competitive", "colluding", "noisy")


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _discretize(values: np.ndarray, bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size <= 2:
        return np.zeros_like(values, dtype=int)
    return np.digitize(values, edges[1:-1])


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left_bin = np.rint(left * 100).astype(int)
    right_bin = np.rint(right * 100).astype(int)
    left_bin -= left_bin.min()
    right_bin -= right_bin.min()
    joint, _, _ = np.histogram2d(
        left_bin,
        right_bin,
        bins=(left_bin.max() + 1, right_bin.max() + 1),
    )
    joint = joint / max(joint.sum(), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(
        np.sum(joint[mask] * np.log2(joint[mask] / (px @ py)[mask]))
    )


def _common_wave(period: int, phase: float) -> float:
    return 0.014 * math.sin(2.0 * math.pi * period / 17.0 + phase) + 0.007 * math.sin(
        2.0 * math.pi * period / 31.0 + phase / 2.0
    )


def _choose_prices(
    regime: str,
    previous: np.ndarray,
    period: int,
    phase: float,
    rng: np.random.Generator,
    punishment: list[int],
    market: dict,
    behavior_noise: float = 0.0,
    coordination_wave_scale: float = 1.0,
) -> np.ndarray:
    if regime in {"competitive", "colluding"} and rng.random() < behavior_noise:
        regime = "colluding" if regime == "competitive" else "competitive"
    if regime == "noisy":
        return rng.uniform(MARKET["p_min"], MARKET["p_max"], len(previous))
    if regime == "competitive":
        chosen = []
        for agent_id in range(len(previous)):
            rivals = np.delete(previous, agent_id)
            chosen.append(max(market["c"], float(np.min(rivals)) - 0.01))
        return np.asarray(chosen)

    target = market["collusive_price"] + (
        _common_wave(period, phase) * coordination_wave_scale
    )
    chosen = []
    for agent_id in range(len(previous)):
        if punishment[agent_id] > 0:
            punishment[agent_id] -= 1
            chosen.append(market["c"])
            continue
        rivals = np.delete(previous, agent_id)
        if rivals.size and float(np.min(rivals)) < market["collusive_price"] - 0.2:
            punishment[agent_id] = market["punishment_rounds"]
            chosen.append(market["c"])
        else:
            chosen.append(target)
    return np.asarray(chosen)


def _market_outcomes(prices: np.ndarray, market: dict) -> tuple[np.ndarray, np.ndarray]:
    utilities = -market["lam"] * prices
    utilities -= utilities.max()
    weights = np.exp(utilities)
    demand = market["M"] * weights / weights.sum()
    return demand, (prices - market["c"]) * demand


def extract_features(
    prices: np.ndarray,
    demand: np.ndarray,
    marginal_cost: float = MARKET["c"],
) -> dict:
    """Extract the 15 detector features from one price/demand path."""

    markup = (prices - marginal_cost) / marginal_cost
    mean_round_markup = markup.mean(axis=1)
    changes = np.diff(prices, axis=0)
    prices_a = prices[:-1, 0]
    prices_b = prices[:-1, 1]
    lead_lag = max(
        abs(_safe_corr(prices[:-1, 0], prices[1:, 1])),
        abs(_safe_corr(prices[:-1, 1], prices[1:, 0])),
    )
    strategy_counts = np.zeros(3)
    for period in range(1, len(prices)):
        for agent_id in range(prices.shape[1]):
            rival = prices[period - 1, 1 - agent_id]
            delta = prices[period, agent_id] - rival
            if delta < -0.05:
                strategy_counts[0] += 1
            elif delta > 0.05:
                strategy_counts[2] += 1
            else:
                strategy_counts[1] += 1
    strategy_prob = strategy_counts / strategy_counts.sum()
    entropy = -sum(
        probability * math.log2(probability)
        for probability in strategy_prob
        if probability > 0
    )
    autocorrelations = []
    for agent_id in range(prices.shape[1]):
        autocorrelations.append(
            _safe_corr(prices[:-1, agent_id], prices[1:, agent_id])
        )
    return {
        "mean_markup": float(mean_round_markup.mean()),
        "max_markup": float(mean_round_markup.max()),
        "markup_persistence": float(np.mean(mean_round_markup > 0.1)),
        "price_variance": float(np.var(prices)),
        "cross_agent_corr": _safe_corr(prices[:, 0], prices[:, 1]),
        "sync_index": float(np.mean(np.ptp(prices, axis=1) < 0.05)),
        "mutual_information": _mutual_information(prices[:, 0], prices[:, 1]),
        "price_dispersion": float(np.mean(np.std(prices, axis=1))),
        "stability": float(1.0 - np.mean(np.abs(changes)) / 2.0),
        "autocorrelation_lag1": float(np.mean(autocorrelations)),
        "volatility": float(np.mean(np.std(changes, axis=0))),
        "undercut_rate": float(strategy_prob[0]),
        "match_rate": float(strategy_prob[1]),
        "strategy_entropy": float(entropy),
        "lead_lag_corr": float(lead_lag),
    }


def run_episode_arrays(
    regime: str,
    seed: int,
    behavior_noise: float = 0.0,
    n_rounds: int | None = None,
    initial_price: float | None = None,
    collusive_price: float | None = None,
    coordination_strength: float = 1.0,
) -> dict:
    """Run one episode and retain its raw arrays for windowed inference."""

    rng = np.random.default_rng(seed)
    market = dict(MARKET)
    if n_rounds is not None:
        market["n_rounds"] = int(n_rounds)
    if collusive_price is not None:
        market["collusive_price"] = float(collusive_price)
    start = (
        market["collusive_price"]
        if initial_price is None
        else float(initial_price)
    )
    prices = np.full(market["n_agents"], start)
    phase = rng.uniform(0, 2 * math.pi)
    punishment = [0] * MARKET["n_agents"]
    price_log = np.empty((market["n_rounds"], market["n_agents"]))
    demand_log = np.empty_like(price_log)
    for period in range(market["n_rounds"]):
        prices = np.clip(
            _choose_prices(
                regime,
                prices,
                period,
                phase,
                rng,
                punishment,
                market,
                behavior_noise,
                coordination_strength,
            ),
            market["p_min"],
            market["p_max"],
        )
        demand, _ = _market_outcomes(prices, market)
        price_log[period] = prices
        demand_log[period] = demand
    return {
        "prices": price_log,
        "demand": demand_log,
        "regime": regime,
        "seed": int(seed),
        "behavior_noise": float(behavior_noise),
        "n_rounds": int(market["n_rounds"]),
        "initial_price": float(start),
        "collusive_price": float(market["collusive_price"]),
        "coordination_strength": float(coordination_strength),
    }


def run_episode(
    regime: str,
    seed: int,
    behavior_noise: float = 0.0,
    n_rounds: int | None = None,
    initial_price: float | None = None,
    collusive_price: float | None = None,
    coordination_strength: float = 1.0,
) -> dict:
    """Run one labeled scripted-market episode."""

    raw = run_episode_arrays(
        regime,
        seed,
        behavior_noise,
        n_rounds=n_rounds,
        initial_price=initial_price,
        collusive_price=collusive_price,
        coordination_strength=coordination_strength,
    )
    features = extract_features(
        raw["prices"],
        raw["demand"],
        MARKET["c"],
    )
    cmi = screen_prices(
        raw["prices"],
        hold_threshold=0.01,
        n_perm=100,
        seed=seed,
    )
    return {
        **features,
        "cmi_bits": cmi["cmi_bits"],
        "cmi_z": cmi["z"],
        "cmi_p_value": cmi["p_value"],
        "cmi_flagged": int(cmi["flagged"]),
        "cmi_top_pair": cmi["top_pair"],
        "cmi_action_entropy": cmi["action_entropy_bits"],
        "cmi_frozen": int(cmi["frozen"]),
        "regime": regime,
        "label": int(regime == "colluding"),
        "seed": int(seed),
        "n_agents": MARKET["n_agents"],
        "n_rounds": raw["n_rounds"],
        "shared_data": True,
        "homogeneous": True,
        "obs_noise": 0.0,
        "behavior_noise": float(behavior_noise),
        "initial_price": raw["initial_price"],
        "collusive_price": raw["collusive_price"],
        "coordination_strength": raw["coordination_strength"],
    }


def run_switch_episode(
    seed: int,
    switch_round: int = 500,
) -> dict:
    """Run a single episode that starts competitive and switches to collusion."""

    rng = np.random.default_rng(seed)
    market = dict(MARKET)
    market["collusive_price"] = 2.0
    prices = np.full(market["n_agents"], market["c"])
    phase = rng.uniform(0, 2 * math.pi)
    punishment = [0] * market["n_agents"]
    price_log = np.empty((market["n_rounds"], market["n_agents"]))
    demand_log = np.empty_like(price_log)
    for period in range(market["n_rounds"]):
        regime = "competitive" if period < switch_round else "colluding"
        if period == switch_round:
            prices = np.full(market["n_agents"], market["collusive_price"])
            punishment = [0] * market["n_agents"]
        prices = np.clip(
            _choose_prices(
                regime,
                prices,
                period,
                phase,
                rng,
                punishment,
                market,
                behavior_noise=0.0,
            ),
            market["p_min"],
            market["p_max"],
        )
        demand, _ = _market_outcomes(prices, market)
        price_log[period] = prices
        demand_log[period] = demand
    return {
        "prices": price_log,
        "demand": demand_log,
        "regime": "competitive_to_colluding",
        "seed": int(seed),
        "behavior_noise": 0.0,
        "n_rounds": market["n_rounds"],
        "switch_round": int(switch_round),
    }


def build_episode_dataset(
    n_per_regime: int = 200,
    train_fraction: float = 0.70,
    training_noise: float = 0.10,
) -> pd.DataFrame:
    """Write 600 episodes with noisy training and clean test partitions."""

    rows = []
    n_train = int(round(n_per_regime * train_fraction))
    for regime_index, regime in enumerate(REGIMES):
        for index in range(n_per_regime):
            is_train = index < n_train
            seed = SEED + regime_index * 10_000 + index
            natural_rng = np.random.default_rng(seed + 900_000)
            coordination_strength = float(natural_rng.uniform(0.0, 1.0))
            collusive_target = float(natural_rng.uniform(1.55, 2.45))
            collusive_target = 1.0 + (
                collusive_target - 1.0
            ) * coordination_strength
            rows.append(
                {
                    **run_episode(
                        regime=regime,
                        seed=seed,
                        behavior_noise=training_noise if is_train else 0.0,
                        n_rounds=int(natural_rng.choice([100, 200, 1000])),
                        initial_price=float(
                            natural_rng.uniform(1.2, 3.0)
                        ),
                        collusive_price=collusive_target,
                        coordination_strength=coordination_strength,
                    ),
                    "split": "train" if is_train else "test",
                    "coordination_strength": coordination_strength,
                }
            )
    frame = pd.DataFrame(rows)
    # Diagnostic composite, excluded from the detector's feature matrix.
    frame["collusion_score"] = (
        0.55 * np.clip(frame["mean_markup"], 0, 1)
        + 0.20 * np.clip(frame["mutual_information"] / 3.0, 0, 1)
        + 0.15 * frame["sync_index"]
        + 0.10 * frame["markup_persistence"]
    )
    frame.to_csv(RESULTS_DIR / "episodes.csv", index=False)
    return frame


if __name__ == "__main__":
    episodes = build_episode_dataset()
    means = episodes.groupby("regime")["mean_markup"].mean()
    print(means)
    assert means["competitive"] < 0.5
    assert means["colluding"] > 0.35
    assert (episodes.groupby("split").size() == [180, 420]).all()
    print("Core episode dataset sanity check passed")
