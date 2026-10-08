"""
Making elicited percentiles acceptable to Metaculus without changing what they say.

* Closed bounds: outcomes beyond the bound are impossible, so such values are clipped
  just inside the bound.
* Open bounds: Metaculus only sees the total mass beyond the bound, so values far outside
  can be squeezed (monotonically) closer to the bound without changing the scored forecast.
* A forecast whose median is more than one full range-width outside the question range is
  almost always a units mistake (e.g. basis points vs percent) and is rejected.
"""

from __future__ import annotations

from nexora.parsing import enforce_increasing


class UnitsMistake(ValueError):
    pass


def _squeeze(values: list[float], start: float, end: float) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return [start + (end - start) * i / (10 * len(values)) for i in range(len(values))]
    return [start + (v - lo) / (hi - lo) * (end - start) for v in values]


def fit_points_to_range(
    points: list[tuple[float, float]],
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
) -> list[tuple[float, float]]:
    if upper <= lower:
        raise ValueError("Invalid question bounds")
    width = upper - lower
    percentiles = [p for p, _ in points]
    values = [v for _, v in points]
    median = min(points, key=lambda pv: abs(pv[0] - 0.5))[1]
    if median < lower - width or median > upper + width:
        raise UnitsMistake(
            f"Median {median} is far outside the question range [{lower}, {upper}] - probably a units mistake"
        )

    inner_margin = 0.002 * width
    if not open_lower:
        values = [max(v, lower + inner_margin) for v in values]
    if not open_upper:
        values = [min(v, upper - inner_margin) for v in values]

    values = [min(max(v, lower - 1.9 * width), upper + 1.9 * width) for v in values]
    if min(values) > upper + 0.2 * width:  # everything above an open upper bound
        values = _squeeze(values, upper + 0.2 * width, upper + 1.8 * width)
    if max(values) < lower - 0.2 * width:  # everything below an open lower bound
        values = _squeeze(values, lower - 1.8 * width, lower - 0.2 * width)

    if zero_point is not None:
        floor = zero_point + 1e-9 * max(width, 1.0)
        values = [max(v, floor) for v in values]

    return enforce_increasing(list(zip(percentiles, values)))
