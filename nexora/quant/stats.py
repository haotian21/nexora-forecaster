"""Small, dependency-free statistics helpers used by the quantitative models."""

from __future__ import annotations

import math
from datetime import date

import numpy as np


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ewma_volatility(changes: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics-style exponentially weighted volatility of the latest observation."""
    if changes.size == 0:
        raise ValueError("No changes to estimate volatility from")
    var = float(np.var(changes[: min(30, changes.size)])) or 1e-12
    for r in changes:
        var = lam * var + (1 - lam) * float(r) ** 2
    return math.sqrt(var)


def blended_volatility(changes: np.ndarray, lam: float = 0.94, recent_weight: float = 0.8) -> float:
    """Mix of current (EWMA) and long-run volatility: reacts to regime changes without
    overreacting to a single noisy week."""
    long_run = float(np.std(changes, ddof=1)) if changes.size > 1 else float(abs(changes[0]))
    recent = ewma_volatility(changes, lam)
    return math.sqrt(recent_weight * recent**2 + (1 - recent_weight) * long_run**2)


def standardized_residuals(changes: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """z_t = r_t / sigma_{t-1}: keeps the fat tails of the data but removes the vol regime."""
    if changes.size < 10:
        raise ValueError("Need at least 10 changes")
    var = float(np.var(changes[: min(30, changes.size)])) or 1e-12
    zs = np.empty(changes.size)
    for i, r in enumerate(changes):
        zs[i] = r / math.sqrt(var) if var > 0 else 0.0
        var = lam * var + (1 - lam) * float(r) ** 2
    zs = zs[5:]  # drop burn-in
    std = float(np.std(zs))
    return zs / std if std > 0 else zs


def simulate_sum_of_changes(
    changes: np.ndarray,
    horizon_steps: int,
    n_sims: int = 40000,
    widen: float = 1.05,
    seed: int = 7,
) -> np.ndarray:
    """Filtered historical simulation of the sum of `horizon_steps` future changes.

    Standardised residuals are bootstrapped and rescaled by today's blended volatility,
    then widened a little for model uncertainty (parameter error, data revisions)."""
    if horizon_steps <= 0:
        return np.zeros(n_sims)
    sigma = blended_volatility(changes)
    zs = standardized_residuals(changes)
    rng = np.random.default_rng(seed)
    draws = rng.choice(zs, size=(n_sims, horizon_steps), replace=True)
    return widen * sigma * draws.sum(axis=1)


def business_days_after(start: date, end: date) -> int:
    """Number of weekdays in (start, end]."""
    if end <= start:
        return 0
    return int(np.busday_count(np.datetime64(start) + 1, np.datetime64(end) + 1))


def percentile_points(samples: np.ndarray, percentiles: list[float]) -> list[tuple[float, float]]:
    """[(fraction, value)] with strictly increasing values."""
    values = np.percentile(samples, [p * 100 for p in percentiles])
    points: list[tuple[float, float]] = []
    span = float(values[-1] - values[0]) or max(abs(float(values[0])), 1.0)
    for p, v in zip(percentiles, values):
        v = float(v)
        if points and v <= points[-1][1]:
            v = points[-1][1] + span * 1e-6
        points.append((p, v))
    return points


QUANT_PERCENTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]
