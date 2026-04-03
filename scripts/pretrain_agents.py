"""Pretrain lightweight agents for deployment."""
from __future__ import annotations

from src.training.train_agents import train_agent


def main() -> None:
    for agent in ("ppo", "sac", "td3"):
        train_agent(agent=agent, total_timesteps=10_000)


if __name__ == "__main__":
    main()
