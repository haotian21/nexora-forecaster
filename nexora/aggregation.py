"""
Turning several forecasts into one.

Binary: trimmed mean on the log-odds scale (a geometric mean of odds), optional blend
with a statistical model, Platt-style recalibration, then a hard cap. Log scores punish
confident misses far more than they reward confident hits, so the cap is cheap insurance.

Multiple choice: linear pool (average) with a probability floor per option.

Numeric/date: linear pool of the 201-point CDFs (averaging CDFs = mixing densities),
which is robust when one forecaster is confidently wrong.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EPS = 1e-4


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def aggregate_binary(probabilities: Sequence[float], trim: bool = True) -> float:
    if not probabilities:
        raise ValueError("No binary forecasts to aggregate")
    for p in probabilities:
        if not (0.0 <= p <= 1.0) or math.isnan(p):
            raise ValueError(f"Invalid probability {p}")
    zs = sorted(logit(min(max(p, 0.001), 0.999)) for p in probabilities)
    if trim and len(zs) >= 5:
        zs = zs[1:-1]
    return sigmoid(sum(zs) / len(zs))


def postprocess_binary(
    p: float,
    quant_probability: float | None = None,
    quant_weight: float = 0.7,
    calib_a: float = 1.0,
    calib_b: float = 0.0,
    floor: float = 0.02,
    ceiling: float = 0.98,
) -> float:
    z = logit(p)
    if quant_probability is not None:
        w = min(max(quant_weight, 0.0), 1.0)
        z = w * logit(quant_probability) + (1 - w) * z
    z = calib_a * z + calib_b
    out = sigmoid(z)
    return float(min(max(out, floor), ceiling))


def apply_floor(probabilities: dict[str, float], floor: float) -> dict[str, float]:
    """Normalise and guarantee every option at least `floor` (mixing with uniform)."""
    n = len(probabilities)
    if n == 0:
        raise ValueError("No options")
    total = sum(max(v, 0.0) for v in probabilities.values())
    if total <= 0:
        return {k: 1.0 / n for k in probabilities}
    floor = min(floor, 0.5 / n)
    scale = 1.0 - n * floor
    return {k: floor + scale * max(v, 0.0) / total for k, v in probabilities.items()}


def aggregate_multiple_choice(
    forecasts: Sequence[dict[str, float]], options: Sequence[str], floor: float = 0.01
) -> dict[str, float]:
    if not forecasts:
        raise ValueError("No multiple-choice forecasts to aggregate")
    pooled = {option: 0.0 for option in options}
    for forecast in forecasts:
        missing = set(options) - set(forecast)
        if missing:
            raise ValueError(f"Forecast is missing options: {missing}")
        normalised = apply_floor({o: forecast[o] for o in options}, 0.0)
        for option in options:
            pooled[option] += normalised[option] / len(forecasts)
    return apply_floor(pooled, floor)


def pool_cdfs(cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> list[float]:
    """Weighted pointwise average of CDF heights evaluated on the same x-grid."""
    if not cdfs:
        raise ValueError("No CDFs to pool")
    arr = np.asarray(cdfs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("CDFs must be a 2D array")
    if weights is None:
        w = np.full(arr.shape[0], 1.0 / arr.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != arr.shape[0] or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid weights")
        w = w / w.sum()
    pooled = (w[:, None] * arr).sum(axis=0)
    pooled = np.clip(pooled, 0.0, 1.0)
    pooled = np.maximum.accumulate(pooled)  # guard against float noise
    return pooled.tolist()


def blend_weights(n_llm: int, quant_weight: float | None) -> list[float]:
    """Weights for [llm_1..llm_n, quant]. Without a quant model all weight is on the LLMs."""
    if n_llm <= 0:
        raise ValueError("Need at least one LLM forecast")
    if quant_weight is None:
        return [1.0 / n_llm] * n_llm
    q = min(max(quant_weight, 0.0), 1.0)
    return [(1.0 - q) / n_llm] * n_llm + [q]
