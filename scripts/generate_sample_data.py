"""Generate realistic synthetic LOB sample data for OPTEX.

This script produces parquet files in data/sample/ for 10 synthetic stocks,
500 episodes each, 30 time steps per episode. Data contains book and trade
columns compatible with downstream feature and environment code.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "HINDUNILVR", "ITC", "KOTAKBANK"]
EPISODES = 500
STEPS = 30
DT = 1.0 / 390  # trading day fraction per step


def ou_process(mu: float, kappa: float, sigma: float, steps: int, s0: float) -> np.ndarray:
    """Simulate Ornstein-Uhlenbeck mid-price path.

    Args:
        mu: Long-run mean level.
        kappa: Mean reversion speed.
        sigma: Volatility parameter.
        steps: Number of time steps.
        s0: Initial price.

    Returns:
        Array of shape (steps,) with mid prices.
    """
    prices = np.zeros(steps)
    prices[0] = s0
    for t in range(1, steps):
        dW = np.random.normal(scale=math.sqrt(DT))
        prices[t] = (
            prices[t - 1]
            + kappa * (mu - prices[t - 1]) * DT
            + sigma * math.sqrt(max(prices[t - 1], 1e-3)) * dW
        )
    return prices


def u_shaped_volume_profile(steps: int) -> np.ndarray:
    """U-shaped intraday volume profile normalized to 1.

    Args:
        steps: Number of time steps.

    Returns:
        Normalized profile array.
    """
    x = np.linspace(0, math.pi, steps)
    profile = 0.6 + 0.4 * np.cos(x)
    profile = profile / profile.sum()
    return profile


def generate_episode(stock: str, episode: int) -> pl.DataFrame:
    """Generate one episode of LOB and trades.

    Args:
        stock: Stock identifier.
        episode: Episode index.

    Returns:
        Polars DataFrame for this episode.
    """
    base_price = 90 + 20 * np.random.rand()
    mu = 100.0
    kappa = 2.0
    sigma = 1.0 + 0.2 * np.random.rand()
    mid = ou_process(mu=mu, kappa=kappa, sigma=sigma, steps=STEPS, s0=base_price)

    spread_bps = np.random.lognormal(mean=math.log(5e-4), sigma=0.4, size=STEPS)
    spread = mid * spread_bps
    ask = mid + spread / 2
    bid = mid - spread / 2

    # depth using Pareto, heavier at L1
    def pareto(alpha: float, size: Tuple[int, ...]) -> np.ndarray:
        return (np.random.pareto(alpha, size) + 1) * 100

    bid_sizes = np.vstack([
        pareto(2.0 + lvl * 0.2, STEPS) for lvl in range(3)
    ]).T
    ask_sizes = np.vstack([
        pareto(2.0 + lvl * 0.2, STEPS) for lvl in range(3)
    ]).T

    # trades
    trade_size = (np.random.pareto(1.5, STEPS) + 1) * 50
    trade_dir = np.random.choice([-1, 1], size=STEPS)
    price_impact = trade_dir * (trade_size / trade_size.max()) * spread
    trade_price = mid + price_impact / 2

    vol_profile = u_shaped_volume_profile(STEPS)
    cum_vol = (vol_profile * trade_size.sum()).cumsum()

    df = pl.DataFrame(
        {
            "stock_id": stock,
            "episode": episode,
            "time_step": np.arange(STEPS),
            "seconds_in_bucket": np.arange(STEPS) * 60,  # 1-min buckets
            "mid": mid,
            "bid_price1": bid,
            "ask_price1": ask,
            "bid_size1": bid_sizes[:, 0],
            "ask_size1": ask_sizes[:, 0],
            "bid_price2": bid * (1 - 5e-4),
            "ask_price2": ask * (1 + 5e-4),
            "bid_size2": bid_sizes[:, 1],
            "ask_size2": ask_sizes[:, 1],
            "bid_price3": bid * (1 - 1e-3),
            "ask_price3": ask * (1 + 1e-3),
            "bid_size3": bid_sizes[:, 2],
            "ask_size3": ask_sizes[:, 2],
            "trade_price": trade_price,
            "trade_size": trade_size,
            "trade_dir": trade_dir,
            "cumulative_volume": cum_vol,
        }
    )
    return df


def generate() -> None:
    """Generate full dataset and save parquet and NumPy helper files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for stock in STOCKS:
        for ep in range(EPISODES):
            frames.append(generate_episode(stock, ep))
    df = pl.concat(frames, how="vertical")
    df.write_parquet(DATA_DIR / "lob.parquet")

    # also save numpy arrays grouped per stock for quick loading
    for stock in STOCKS:
        sdf = df.filter(pl.col("stock_id") == stock).sort(["episode", "time_step"])
        np.savez(
            DATA_DIR / f"{stock}_episodes.npz",
            mid=sdf["mid"].to_numpy().reshape(EPISODES, STEPS),
            bid=sdf["bid_price1"].to_numpy().reshape(EPISODES, STEPS),
            ask=sdf["ask_price1"].to_numpy().reshape(EPISODES, STEPS),
            trade_price=sdf["trade_price"].to_numpy().reshape(EPISODES, STEPS),
            trade_size=sdf["trade_size"].to_numpy().reshape(EPISODES, STEPS),
            trade_dir=sdf["trade_dir"].to_numpy().reshape(EPISODES, STEPS),
        )

    print(f"Generated synthetic LOB data at {DATA_DIR}")


if __name__ == "__main__":
    generate()
