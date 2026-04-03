"""Custom Gymnasium environment for optimal execution."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from src.data.loader import DataLoader
from src.data.features import compute_features
from src.environment.reward_shaping import compute_reward
from src.models.impact_models import impact, ImpactType


class LOBExecutionEnv(gym.Env):
    """Limit order book execution environment.

    Observation: 16-dim vector of microstructure signals.
    Action: fraction of remaining inventory to execute in this step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_path: str | None = None,
        impact_kind: ImpactType = "linear",
        risk_lambda: float = 1e-5,
        horizon: int = 30,
        inventory: float = 50_000,
        adv: float = 1_000_000,
        seed: int | None = 7,
    ) -> None:
        super().__init__()
        self.loader = DataLoader(Path(data_path) if data_path else None)
        self.impact_kind = impact_kind
        self.risk_lambda = risk_lambda
        self.horizon = horizon
        self.initial_inventory = inventory
        self.adv = adv
        self.seed(seed)

        high = np.array([np.inf] * 16, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._data = self._prepare_data()
        self._current_df: pl.DataFrame | None = None
        self._t = 0
        self.inventory = float(inventory)
        self.arrival_price = 0.0

    def _prepare_data(self) -> Dict[Tuple[str, int], pl.DataFrame]:
        lf = compute_features(self.loader.load())
        df = lf.collect()
        grouped: Dict[Tuple[str, int], pl.DataFrame] = {}
        for (sid, ep), sub in df.groupby(["stock_id", "episode"], maintain_order=True):
            grouped[(sid, int(ep))] = sub.sort("time_step")
        return grouped

    def seed(self, seed: int | None = None) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def _sample_episode(self) -> pl.DataFrame:
        key = random.choice(list(self._data.keys()))
        return self._data[key]

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._current_df = self._sample_episode()
        self._t = 0
        self.inventory = float(self.initial_inventory)
        self.arrival_price = float(self._current_df["mid_price"][0])
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        assert self._current_df is not None
        row = self._current_df.row(self._t)
        obs = np.array(
            [
                self.inventory / self.initial_inventory,
                self._t / self.horizon,
                row[self._current_df.columns.index("mid_price")] - self.arrival_price,
                row[self._current_df.columns.index("spread_bps")],
                row[self._current_df.columns.index("queue_imbalance")],
                row[self._current_df.columns.index("depth_ratio_3l")],
                row[self._current_df.columns.index("realized_vol")],
                row[self._current_df.columns.index("kyle_lambda")],
                row[self._current_df.columns.index("amihud")],
                row[self._current_df.columns.index("tick_dir")],
                row[self._current_df.columns.index("volume_clock")],
                row[self._current_df.columns.index("mom_5")],
                row[self._current_df.columns.index("mom_10")],
                row[self._current_df.columns.index("mom_30")],
                getattr(self, "last_action", 0.0),
                getattr(self, "steps_since_trade", 0),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: np.ndarray):  # type: ignore[override]
        assert self._current_df is not None
        frac = float(np.clip(action[0], 0.0, 1.0))
        vol_to_trade = frac * self.inventory
        row = self._current_df.row(self._t)
        mid = float(row[self._current_df.columns.index("mid_price")])
        spread_bps = float(row[self._current_df.columns.index("spread_bps")])

        exec_price = mid - impact(vol_to_trade, self.adv, eta=1e-4, kind=self.impact_kind)
        self.inventory -= vol_to_trade
        self.last_action = frac
        self.steps_since_trade = 0 if vol_to_trade > 0 else getattr(self, "steps_since_trade", 0) + 1

        time_remaining = max(0.0, (self.horizon - self._t) / self.horizon)
        reward = compute_reward(
            exec_price=exec_price,
            arrival_price=self.arrival_price,
            volume=vol_to_trade,
            spread_bps=spread_bps,
            mid_price=mid,
            inventory=self.inventory,
            risk_lambda=self.risk_lambda,
            realized_vol=float(row[self._current_df.columns.index("realized_vol")]),
            time_remaining=time_remaining,
        )

        self._t += 1
        terminated = self._t >= self.horizon or self.inventory <= 1e-6
        if terminated and self.inventory > 1e-6:
            reward -= 10.0 * self.inventory  # urgency liquidation
            self.inventory = 0.0
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()
        info = {"remaining_inventory": self.inventory}
        return obs, reward, terminated, False, info

    def render(self, mode: str = "human") -> None:
        pass


__all__ = ["LOBExecutionEnv"]
