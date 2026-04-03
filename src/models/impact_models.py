"""Market impact model primitives."""
from __future__ import annotations

from typing import Literal

import numpy as np

ImpactType = Literal["linear", "sqrt", "power"]


def impact(volume: float, adv: float, eta: float, kind: ImpactType) -> float:
    """Compute impact in price units for a single trade.

    Args:
        volume: Trade volume.
        adv: Average daily volume.
        eta: Impact coefficient.
        kind: Impact functional form: linear, sqrt, or power.

    Returns:
        Impact value.
    """
    x = volume / (adv + 1e-6)
    if kind == "linear":
        return eta * x
    if kind == "sqrt":
        return eta * np.sign(x) * np.sqrt(abs(x))
    if kind == "power":
        return eta * (abs(x) ** 0.6) * np.sign(x)
    raise ValueError(f"Unknown impact kind {kind}")


def vectorized_impact(volumes: np.ndarray, adv: float, eta: float, kind: ImpactType) -> np.ndarray:
    """Vectorized version of impact for arrays."""
    return np.array([impact(v, adv, eta, kind) for v in volumes])


__all__ = ["impact", "vectorized_impact", "ImpactType"]
