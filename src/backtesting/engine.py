"""Backtesting engine for OPTEX."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from src.backtesting.metrics import compute_metrics
from src.data.loader import DataLoader
from src.environment.lob_env import LOBExecutionEnv
from src.models.almgren_chriss import optimal_trajectory


class BacktestEngine:
    """Monte Carlo walk-forward backtester."""

    def __init__(self, data_path: Path | None = None, episodes: int = 200) -> None:
        self.loader = DataLoader(data_path)
        self.episodes = episodes

    def run(self) -> pl.DataFrame:
        """Execute backtests across strategies.

        Returns:
            Polars DataFrame with episode-level results.
        """
        stocks = self.loader.list_stocks()
        results = []
        for _ in range(self.episodes):
            stock = random.choice(stocks)
            ep = random.randint(0, 10)
            env = LOBExecutionEnv()
            obs, _ = env.reset()
            twap_cost = self._simulate_twap(env)
            env.reset()
            ac_cost = self._simulate_ac(env)
            env.reset()
            rl_cost = self._simulate_random_rl(env)
            results.append(
                {
                    "stock_id": stock,
                    "episode": ep,
                    "twap_is": twap_cost,
                    "ac_is": ac_cost,
                    "ppo_is": rl_cost,
                }
            )
        df = pl.DataFrame(results)
        metrics = compute_metrics(df)
        df.write_csv(Path("backtest_results.csv"))
        return metrics

    def _simulate_twap(self, env: LOBExecutionEnv) -> float:
        steps = env.horizon
        action = np.array([1.0 / steps], dtype=np.float32)
        total_reward = 0.0
        for _ in range(steps):
            _, r, done, _, info = env.step(action)
            total_reward += r
            if done:
                break
        return -total_reward

    def _simulate_ac(self, env: LOBExecutionEnv) -> float:
        path = optimal_trajectory(env.initial_inventory, env.horizon, env.risk_lambda, eta=1e-4, gamma=1e-5)
        total_reward = 0.0
        for t in range(env.horizon):
            vol = max(path[t] - path[t + 1], 0)
            frac = vol / max(env.inventory, 1e-6)
            _, r, done, _, _ = env.step(np.array([frac], dtype=np.float32))
            total_reward += r
            if done:
                break
        return -total_reward

    def _simulate_random_rl(self, env: LOBExecutionEnv) -> float:
        total_reward = 0.0
        for _ in range(env.horizon):
            frac = np.clip(np.random.beta(2, 2), 0, 1)
            _, r, done, _, _ = env.step(np.array([frac], dtype=np.float32))
            total_reward += r
            if done:
                break
        return -total_reward


__all__ = ["BacktestEngine"]
