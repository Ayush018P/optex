"""Almgren-Chriss optimal execution utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ACParams:
    eta: float  # temporary impact coefficient
    gamma: float  # permanent impact coefficient
    sigma: float  # volatility of returns


def optimal_trajectory(q0: float, T: int, lam: float, eta: float, gamma: float, dt: float = 1.0) -> np.ndarray:
    """Compute optimal deterministic trajectory using AC closed form.

    Args:
        q0: Initial inventory (positive for sell).
        T: Number of periods.
        lam: Risk aversion parameter.
        eta: Temporary impact.
        gamma: Permanent impact.
        dt: Time step length.

    Returns:
        Array of size T+1 with remaining inventory over time.
    """
    tau = math.sqrt(lam * sigma_sq_est(eta, dt) / eta)
    kappa = math.acosh(1 + (tau ** 2) / 2)
    times = np.arange(T + 1)
    denom = math.sinh(kappa * T)
    path = q0 * np.sinh(kappa * (T - times)) / denom
    path[-1] = 0.0
    return path


def sigma_sq_est(eta: float, dt: float) -> float:
    """Rough volatility proxy used in AC formulas.

    Args:
        eta: Temporary impact.
        dt: Time step.

    Returns:
        Volatility estimate.
    """
    return max(1e-6, eta * dt)


def expected_cost(q0: float, path: np.ndarray, eta: float, gamma: float, mid0: float, dt: float = 1.0) -> float:
    """Expected execution cost for a given path.

    Args:
        q0: Initial inventory.
        path: Remaining inventory path length T+1.
        eta: Temporary impact.
        gamma: Permanent impact.
        mid0: Initial mid price.
        dt: Time step.

    Returns:
        Expected implementation shortfall in dollars.
    """
    trades = -np.diff(path)
    temp_cost = eta * np.sum((trades / dt) ** 2) * dt
    perm_cost = gamma * np.sum(np.cumsum(trades))
    return mid0 * perm_cost + temp_cost


def cost_variance(q0: float, path: np.ndarray, sigma: float, dt: float = 1.0) -> float:
    """Variance of cost.

    Args:
        q0: Initial inventory.
        path: Remaining inventory path.
        sigma: Volatility.
        dt: Time step.

    Returns:
        Variance of implementation shortfall.
    """
    trades = -np.diff(path)
    return (sigma ** 2) * np.sum((np.cumsum(trades) ** 2) * dt)


def efficient_frontier(q0: float, T: int, eta: float, gamma: float, mid0: float, sigmas: np.ndarray, lams: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cost/variance pairs across risk aversion grid.

    Args:
        q0: Initial inventory.
        T: Horizon steps.
        eta: Temporary impact.
        gamma: Permanent impact.
        mid0: Initial mid price.
        sigmas: Volatility candidates.
        lams: Risk aversion grid.
        dt: Time step.

    Returns:
        Tuple of arrays (costs, vars).
    """
    costs, vars_ = [], []
    for lam in lams:
        path = optimal_trajectory(q0, T, lam, eta, gamma, dt)
        costs.append(expected_cost(q0, path, eta, gamma, mid0, dt))
        vars_.append(cost_variance(q0, path, sigmas.mean(), dt))
    return np.array(costs), np.array(vars_)


def calibrate_eta_gamma(returns: np.ndarray, volumes: np.ndarray) -> ACParams:
    """Calibrate AC parameters via simple moment matching.

    Args:
        returns: Array of price returns.
        volumes: Executed volumes.

    Returns:
        ACParams with eta and gamma estimates.
    """
    vol_var = np.var(returns)
    vol_mean = np.mean(np.abs(returns))
    volume_scale = np.mean(volumes) + 1e-6
    eta = vol_mean / (volume_scale)
    gamma = vol_var / (volume_scale ** 2)
    sigma = np.std(returns)
    return ACParams(eta=eta, gamma=gamma, sigma=sigma)


__all__ = [
    "optimal_trajectory",
    "expected_cost",
    "cost_variance",
    "efficient_frontier",
    "calibrate_eta_gamma",
    "ACParams",
]
