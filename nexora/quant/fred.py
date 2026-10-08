"""
FRED data-series questions (MiniBench: "What will the value of FRED series DGS5 be?").

Downloads the series (keyless CSV endpoint, or the official API if FRED_API_KEY is set)
and simulates the value on the target date with filtered historical simulation:
bootstrapped standardised daily changes, rescaled to today's volatility.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from nexora.quant.stats import (
    QUANT_PERCENTILES,
    business_days_after,
    percentile_points,
    simulate_sum_of_changes,
)

logger = logging.getLogger(__name__)

_SERIES_PATTERNS = [
    re.compile(r"fred\.stlouisfed\.org/series/([A-Za-z0-9_]{2,40})", re.IGNORECASE),
    re.compile(r"fred\.stlouisfed\.org/graph/\?id=([A-Za-z0-9_]{2,40})", re.IGNORECASE),
    re.compile(r"\bFRED\s+series\s+[\"'“]?([A-Z0-9_]{2,40})\b"),
    re.compile(r"\(FRED(?:\s+series)?(?:\s+ID)?[:\s]+([A-Z0-9_]{2,40})\)"),
]
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
USER_AGENT = "Mozilla/5.0 (compatible; nexora-forecaster/0.1; +https://github.com)"


@dataclass(frozen=True)
class FredSpec:
    series_id: str
    target_date: date


def detect(texts: dict[str, str], scheduled_resolution: datetime | None) -> FredSpec | None:
    joined = "\n".join(v for v in texts.values() if v)
    series_id = None
    for pattern in _SERIES_PATTERNS:
        match = pattern.search(joined)
        if match:
            series_id = match.group(1).upper()
            break
    if not series_id:
        return None
    title = texts.get("question_text") or ""
    dates = _DATE.findall(title)
    if dates:
        target = date.fromisoformat(dates[-1])
    elif scheduled_resolution is not None:
        target = scheduled_resolution.astimezone(timezone.utc).date()
    else:
        return None
    return FredSpec(series_id=series_id, target_date=target)


def fetch_series(series_id: str, timeout: int = 25, years: int = 4) -> list[tuple[date, float]]:
    start = (datetime.now(timezone.utc).date() - timedelta(days=365 * years)).isoformat()
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if api_key:
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start},
            timeout=timeout,
        )
        response.raise_for_status()
        rows = [(o["date"], o["value"]) for o in response.json().get("observations", [])]
    else:
        response = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": series_id, "cosd": start},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        reader = csv.reader(io.StringIO(response.text))
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Unexpected FRED CSV header: {header}")
        rows = [(r[0], r[1]) for r in reader if len(r) >= 2]
    return parse_rows(rows)


def parse_rows(rows: list[tuple[str, str]]) -> list[tuple[date, float]]:
    out: list[tuple[date, float]] = []
    for raw_date, raw_value in rows:
        raw_value = (raw_value or "").strip()
        if raw_value in {"", ".", "NaN", "nan", "#N/A"}:
            continue
        try:
            out.append((date.fromisoformat(raw_date.strip()[:10]), float(raw_value)))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


@dataclass
class FredForecast:
    points: list[tuple[float, float]]  # (percentile fraction, value)
    summary: str
    last_date: date
    last_value: float
    horizon_steps: int


def _frequency(dates: list[date]) -> tuple[str, float]:
    gaps = np.diff([d.toordinal() for d in dates[-120:]])
    median_gap = float(np.median(gaps)) if gaps.size else 1.0
    if median_gap <= 3:
        weekend_share = sum(1 for d in dates[-120:] if d.weekday() >= 5) / max(len(dates[-120:]), 1)
        # Series that also publish on weekends (e.g. crypto prices) step every calendar day.
        return ("calendar-daily" if weekend_share > 0.1 else "daily"), median_gap
    if median_gap <= 9:
        return "weekly", median_gap
    if median_gap <= 35:
        return "monthly", median_gap
    if median_gap <= 100:
        return "quarterly", median_gap
    return "other", median_gap


def horizon_steps(frequency: str, median_gap: float, last: date, target: date) -> int:
    if target <= last:
        return 0
    if frequency == "daily":
        return business_days_after(last, target)
    days = (target - last).days
    if frequency == "calendar-daily":
        return days
    return max(0, int(math.floor(days / median_gap + 1e-9)))


def forecast(spec: FredSpec, observations: list[tuple[date, float]], today: date | None = None) -> FredForecast:
    if len(observations) < 40:
        raise ValueError(f"Not enough FRED observations for {spec.series_id}: {len(observations)}")
    today = today or datetime.now(timezone.utc).date()
    dates = [d for d, _ in observations]
    values = np.array([v for _, v in observations], dtype=float)
    frequency, median_gap = _frequency(dates)
    last_date, last_value = dates[-1], float(values[-1])

    if spec.target_date <= last_date:
        known = [v for d, v in observations if d <= spec.target_date]
        anchor = float(known[-1]) if known else last_value
        spread = max(abs(anchor) * 0.002, 1e-6)
        samples = anchor + np.random.default_rng(1).normal(0, spread, 20000)
        steps = 0
        method = "target date already observed; distribution centred on the published value"
    else:
        steps = horizon_steps(frequency, median_gap, last_date, spec.target_date)
        window = values[-(750 if frequency in {"daily", "calendar-daily"} else 260):]
        use_log = bool(np.all(window > 0))
        changes = np.diff(np.log(window)) if use_log else np.diff(window)
        changes = changes[np.isfinite(changes)]
        if changes.size < 30:
            raise ValueError("Too few changes to model")
        simulated = simulate_sum_of_changes(changes, max(steps, 1))
        samples = last_value * np.exp(simulated) if use_log else last_value + simulated
        method = (
            f"filtered historical simulation of {max(steps, 1)} future {frequency} "
            f"{'log-' if use_log else ''}changes (bootstrapped from {changes.size} past changes, "
            "rescaled to a blend of current EWMA (80%) and long-run volatility (20%), widened 5%)"
        )

    points = percentile_points(samples, QUANT_PERCENTILES)
    one_year_ago = last_date - timedelta(days=365)
    last_year = [v for d, v in observations if d >= one_year_ago]

    def change_since(days: int) -> str:
        ref_date = last_date - timedelta(days=days)
        past = [v for d, v in observations if d <= ref_date]
        return f"{last_value - past[-1]:+.4g}" if past else "n/a"

    recent = ", ".join(f"{d.isoformat()}: {v:g}" for d, v in observations[-8:])
    lookup = dict(points)
    summary = (
        f"FRED series {spec.series_id} ({frequency} data). Last observation {last_date.isoformat()} = {last_value:g}. "
        f"Target date {spec.target_date.isoformat()} is {steps} {frequency} observation(s) ahead (today {today.isoformat()}).\n"
        f"Recent values: {recent}.\n"
        f"Change over 1 week: {change_since(7)}, 1 month: {change_since(30)}, 3 months: {change_since(91)}. "
        f"12-month range: {min(last_year):g} to {max(last_year):g}.\n"
        f"Statistical model ({method}):\n"
        f"  P1={lookup[0.01]:.4g}  P5={lookup[0.05]:.4g}  P10={lookup[0.10]:.4g}  P25={lookup[0.25]:.4g}  "
        f"P50={lookup[0.50]:.4g}  P75={lookup[0.75]:.4g}  P90={lookup[0.90]:.4g}  P95={lookup[0.95]:.4g}  P99={lookup[0.99]:.4g}"
    )
    return FredForecast(points=points, summary=summary, last_date=last_date, last_value=last_value, horizon_steps=steps)
