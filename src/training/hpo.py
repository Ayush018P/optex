"""Optuna hyperparameter search for SAC."""
from __future__ import annotations

import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from src.environment.lob_env import LOBExecutionEnv


def objective(trial: optuna.Trial) -> float:
    """Optuna objective returning mean reward."""
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int("buffer_size", 50_000, 200_000)

    env = make_vec_env(lambda: LOBExecutionEnv(), n_envs=2)
    model = SAC("MlpPolicy", env, gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, verbose=0)
    model.learn(20_000)
    reward = float(model.logger.name_to_value["rollout/ep_rew_mean"]) if model.logger.name_to_value else 0.0
    return reward


def run_study(n_trials: int = 10) -> optuna.Study:
    """Run the Optuna study."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


if __name__ == "__main__":
    study = run_study(5)
    print(study.best_trial)
