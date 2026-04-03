"""Training pipelines for PPO, SAC, and TD3 agents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from src.environment.lob_env import LOBExecutionEnv

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)


def make_env(seed: int, impact_kind: str) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = LOBExecutionEnv(impact_kind=impact_kind, seed=seed + np.random.randint(0, 10))
        return env

    return _init


def train_agent(agent: str = "ppo", total_timesteps: int = 100_000, impact_kind: str = "linear") -> Path:
    """Train specified RL agent and persist weights.

    Args:
        agent: One of ppo, sac, td3.
        total_timesteps: Training steps.
        impact_kind: Impact model to use in env.

    Returns:
        Path to saved model zip.
    """
    n_envs = 4
    vec_env = SubprocVecEnv([make_env(i, impact_kind) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env = VecNormalize(SubprocVecEnv([make_env(100 + i, impact_kind) for i in range(2)]), training=False)

    algo_map = {
        "ppo": PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=str(ROOT / "tb")),
        "sac": SAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=str(ROOT / "tb")),
        "td3": TD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=str(ROOT / "tb")),
    }
    model = algo_map[agent]

    checkpoint_callback = CheckpointCallback(save_freq=10_000 // n_envs, save_path=str(MODELS_DIR), name_prefix=agent)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(MODELS_DIR / f"{agent}_best"), eval_freq=10_000 // n_envs)

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model_path = MODELS_DIR / f"{agent}_final.zip"
    model.save(model_path)
    return model_path


def main() -> None:
    for agent in ("ppo", "sac", "td3"):
        print(f"Training {agent}...")
        path = train_agent(agent=agent, total_timesteps=50_000, impact_kind="linear")
        print(f"Saved {agent} to {path}")


if __name__ == "__main__":
    main()
