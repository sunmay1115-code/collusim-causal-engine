"""Calibrated proxy agents used by Collusim."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibratedLLMProxy:
    """A deterministic-seeded proxy using OLS-extracted parameters."""

    agent_id: int
    params: dict
    market: dict
    seed: int
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 10_007 * self.agent_id)

    def choose_price(
        self,
        observed_competitor_price: float,
        own_last_price: float,
        blind: bool = False,
    ) -> float:
        """Choose a price from the calibrated linear policy."""

        if blind:
            # The signal is unavailable. The previous own price is used as the
            # state estimate, with a small cost-ward correction for stability.
            distance = own_last_price - self.market["c"]
            observed_competitor_price = (
                self.market["c"]
                + (1.0 - self.market["blind_competitive_pull"]) * distance
            )
        residuals = self.params.get("residuals")
        if residuals is None or len(residuals) == 0:
            raise ValueError("Empirical residual distribution is unavailable")
        noise = float(self.rng.choice(residuals))
        target = self.params["intercept"] + self.params["beta"] * (
            observed_competitor_price
        ) + noise
        return float(
            np.clip(target, self.market["p_min"], self.market["p_max"])
        )


@dataclass
class RuleBasedHumanAgent:
    """A non-AI comparator representing a hard operational ban."""

    agent_id: int
    market: dict
    seed: int
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 20_011 * self.agent_id)

    def choose_price(
        self,
        observed_competitor_price: float,
        own_last_price: float,
        blind: bool = False,
    ) -> float:
        del observed_competitor_price, own_last_price, blind
        target = (
            self.market["c"]
            + self.market["human_markup"]
            + self.rng.normal(0.0, self.market["human_price_noise"])
        )
        return float(
            np.clip(target, self.market["p_min"], self.market["p_max"])
        )

