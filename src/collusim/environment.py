"""Repeated pricing environment with calibrated proxy agents."""

from __future__ import annotations

import json
import math

import numpy as np

from config import COLLUSIM_MARKET, RESULTS_DIR
from src.collusim.agents import CalibratedLLMProxy, RuleBasedHumanAgent


def _load_params(params: dict | None = None) -> dict:
    if params is not None:
        return params
    path = RESULTS_DIR / "collusim_params.json"
    if not path.exists():
        raise FileNotFoundError(
            "results/collusim_params.json missing; run calibration first"
        )
    payload = json.loads(path.read_text(encoding="ascii"))
    calibrated = dict(payload["default"])
    residual_path = RESULTS_DIR / "collusim_residuals.npz"
    if not residual_path.exists():
        raise FileNotFoundError(
            "results/collusim_residuals.npz missing; run calibration first"
        )
    with np.load(residual_path) as residual_data:
        calibrated["residuals"] = np.asarray(
            residual_data[payload.get("residual_key", "pooled")],
            dtype=float,
        )
    return calibrated


def _softmax_prices(prices: np.ndarray, market: dict) -> np.ndarray:
    utilities = -market["lam"] * prices
    utilities -= float(np.max(utilities))
    weights = np.exp(utilities)
    return weights / float(weights.sum())


def _near_competitive_price(market: dict) -> float:
    """Competitive reference price used for the one-round history reset."""

    price = market["history_reset_price"]
    return float(np.clip(price, market["p_min"], market["p_max"]))


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.sum(np.exp(values - peak))))


def _welfare(prices: np.ndarray, demand: np.ndarray, market: dict) -> float:
    consumer = (market["M"] / market["lam"]) * _logsumexp(
        market["lam"] * (market["consumer_value"] - prices)
    )
    producer = float(np.sum((prices - market["c"]) * demand))
    return consumer + producer


def run_episode(
    params: dict | None = None,
    seed: int = 0,
    precision: float = 0.0,
    delay: int = 0,
    visibility_loss: float = 0.0,
    history_reset: bool = False,
    reset_round: int = 50,
    agent_type: str = "llm",
) -> dict:
    """Run one Collusim episode.

    visibility_loss is the probability that the competitor signal is missing.
    Thus 0 is perfect information and 1 is completely blind.
    history_reset clears visible price history once and injects one
    near-Nash competitive observation at the configured round.
    """

    calibrated = _load_params(params)
    market = dict(COLLUSIM_MARKET)
    n_agents = market["n_agents"]
    n_rounds = market["n_rounds"]
    if reset_round < 1:
        raise ValueError("reset_round must be at least 1")
    reset_round = int(reset_round)
    reset_applied = False
    rng = np.random.default_rng(seed)
    if agent_type == "human":
        agents = [
            RuleBasedHumanAgent(
                agent_id=index,
                market=market,
                seed=seed,
            )
            for index in range(n_agents)
        ]
    else:
        agents = [
            CalibratedLLMProxy(
                agent_id=index,
                params=calibrated,
                market=market,
                seed=seed,
            )
            for index in range(n_agents)
        ]
    initial = np.full(n_agents, market["initial_price"], dtype=float)
    history = [initial.copy()]
    prices_log = np.empty((n_rounds, n_agents), dtype=float)
    demand_log = np.empty((n_rounds, n_agents), dtype=float)
    profit_log = np.empty((n_rounds, n_agents), dtype=float)
    welfare_log = np.empty(n_rounds, dtype=float)
    blind_log = np.zeros((n_rounds, n_agents), dtype=bool)
    nash_price = _near_competitive_price(market)

    for period in range(n_rounds):
        current_round = period + 1
        inject_reset = bool(
            history_reset
            and agent_type == "llm"
            and current_round == reset_round
        )
        if inject_reset:
            history = []
            prices = np.full(n_agents, nash_price, dtype=float)
            reset_applied = True
        else:
            chosen = np.empty(n_agents, dtype=float)
            own_last = history[-1]
            delayed_index = max(0, len(history) - 1 - int(delay))
            delayed_prices = history[delayed_index]
            precision_reliability = 1.0 / (1.0 + 4.0 * precision)
            delay_reliability = 1.0 / (1.0 + 0.15 * delay)
            for agent_id in range(n_agents):
                rivals = np.delete(delayed_prices, agent_id)
                observed = (
                    float(np.mean(rivals))
                    if rivals.size
                    else own_last[agent_id]
                )
                observed += rng.laplace(0.0, precision)
                observed = market["c"] + precision_reliability * (
                    observed - market["c"]
                )
                observed = market["c"] + delay_reliability * (
                    observed - market["c"]
                )
                blind = bool(rng.random() < visibility_loss)
                blind_log[period, agent_id] = blind
                chosen[agent_id] = agents[agent_id].choose_price(
                    observed_competitor_price=observed,
                    own_last_price=float(own_last[agent_id]),
                    blind=blind,
                )
            prices = np.clip(chosen, market["p_min"], market["p_max"])
        liquidity = (
            market["human_liquidity"] if agent_type == "human" else 1.0
        )
        demand = market["M"] * liquidity * _softmax_prices(prices, market)
        profit = (prices - market["c"]) * demand
        history.append(prices)
        prices_log[period] = prices
        demand_log[period] = demand
        profit_log[period] = profit
        welfare_log[period] = _welfare(prices, demand, market)

    return {
        "prices": prices_log,
        "demand": demand_log,
        "profits": profit_log,
        "welfare": welfare_log,
        "mean_markup": float(
            np.mean((prices_log - market["c"]) / market["c"])
        ),
        "total_welfare": float(np.mean(welfare_log)),
        "blind_share": float(blind_log.mean()),
        "history_reset_applied": reset_applied,
        "history_reset_round": reset_round,
        "history_reset_price": nash_price,
        "params": {
            "seed": int(seed),
            "precision": float(precision),
            "delay": int(delay),
            "visibility_loss": float(visibility_loss),
            "history_reset": bool(history_reset),
            "reset_round": reset_round,
            "agent_type": agent_type,
        },
    }


if __name__ == "__main__":
    perfect = run_episode(seed=7, visibility_loss=0.0)
    blind = run_episode(seed=7, visibility_loss=1.0)
    print(f"perfect-information markup: {perfect['mean_markup']:.4f}")
    print(f"blind markup: {blind['mean_markup']:.4f}")
    assert blind["mean_markup"] < perfect["mean_markup"]
    print("Collusim visibility sanity check passed")
