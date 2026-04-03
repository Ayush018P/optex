import numpy as np
from src.environment.lob_env import LOBExecutionEnv


def test_inventory_conservation():
    env = LOBExecutionEnv()
    obs, _ = env.reset()
    total = env.initial_inventory
    executed = 0
    for _ in range(env.horizon):
        action = np.array([1.0 / env.horizon], dtype=np.float32)
        _, _, done, _, info = env.step(action)
        executed = env.initial_inventory - info["remaining_inventory"]
        if done:
            break
    assert abs(executed - total) < 1e-3


def test_no_negative_inventory():
    env = LOBExecutionEnv()
    env.reset()
    for _ in range(env.horizon):
        _, _, done, _, info = env.step(np.array([1.0], dtype=np.float32))
        assert info["remaining_inventory"] >= 0
        if done:
            break
