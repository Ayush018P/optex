"""Performance metrics for backtests."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import wilcoxon


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient of an array."""
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def compute_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Compute summary metrics vs TWAP baseline."""
    metrics = []
    for col in [c for c in df.columns if c.endswith("_is") and c != "twap_is"]:
        is_vals = df[col].to_numpy()
        twap = df["twap_is"].to_numpy()
        sharpe = -np.mean(is_vals) / (np.std(is_vals) + 1e-6)
        win = float(np.mean(is_vals < twap))
        cvar = float(np.mean(np.sort(is_vals)[int(0.95 * len(is_vals)) :]))
        pval = float(wilcoxon(twap, is_vals).pvalue) if len(is_vals) > 10 else 1.0
        metrics.append(
            {
                "strategy": col.replace("_is", ""),
                "mean_is": float(np.mean(is_vals)),
                "sharpe": float(sharpe),
                "win_rate_vs_twap": win,
                "cvar_95": cvar,
                "p_value": pval,
            }
        )
    return pl.DataFrame(metrics)


__all__ = ["compute_metrics", "gini_coefficient"]
