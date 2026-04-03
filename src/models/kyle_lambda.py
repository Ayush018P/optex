"""Kyle's lambda estimator."""
from __future__ import annotations

import numpy as np
import polars as pl


def estimate_lambda(prices: np.ndarray, signed_volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling OLS Kyle's lambda estimator.

    Args:
        prices: Array of trade prices.
        signed_volume: Signed volume array.
        window: Rolling window length.

    Returns:
        Lambda estimates per time step.
    """
    lam = np.zeros_like(prices)
    for t in range(len(prices)):
        start = max(0, t - window + 1)
        y = np.diff(prices[start : t + 1])
        x = signed_volume[start + 1 : t + 1]
        if len(x) < 2 or np.all(x == 0):
            lam[t] = lam[t - 1] if t > 0 else 0
            continue
        beta = np.dot(x, y) / (np.dot(x, x) + 1e-6)
        lam[t] = beta
    return lam


def estimate_on_frame(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add Kyle lambda column to a Polars frame grouped by stock/episode."""
    out = []
    for (_sid, _ep), sub in df.groupby(["stock_id", "episode"], maintain_order=True):
        lam = estimate_lambda(sub["trade_price"].to_numpy(), sub["trade_size"].to_numpy() * sub["trade_dir"].to_numpy(), window)
        sub = sub.with_columns(pl.Series("kyle_lambda_est", lam))
        out.append(sub)
    return pl.concat(out)


__all__ = ["estimate_lambda", "estimate_on_frame"]
