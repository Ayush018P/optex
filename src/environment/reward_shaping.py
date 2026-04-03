"""Reward shaping utilities for LOB environment."""
from __future__ import annotations

import numpy as np


def compute_reward(
    exec_price: float,
    arrival_price: float,
    volume: float,
    spread_bps: float,
    mid_price: float,
    inventory: float,
    risk_lambda: float,
    realized_vol: float,
    time_remaining: float,
    urgency_penalty: float = 10.0,
) -> float:
    """Compute shaped reward.

    Args:
        exec_price: Execution price for the trade.
        arrival_price: Arrival mid price.
        volume: Volume executed this step.
        spread_bps: Spread in basis points.
        mid_price: Current mid price.
        inventory: Remaining inventory after trade.
        risk_lambda: Risk aversion parameter.
        realized_vol: Realized volatility estimate.
        time_remaining: Fraction of horizon remaining.
        urgency_penalty: Penalty multiplier for leftover inventory.

    Returns:
        Reward (negative cost).
    """
    is_cost = (arrival_price - exec_price) * volume
    spread_cost = (spread_bps / 10000) * mid_price * volume / 2
    risk_cost = risk_lambda * (inventory ** 2) * (realized_vol ** 2)
    urgency = 0.0
    if time_remaining < 0.05 and inventory > 0:
        urgency = urgency_penalty * inventory
    reward = -(is_cost + spread_cost + risk_cost + urgency)
    return float(reward)


__all__ = ["compute_reward"]
