"""
Meta-questions about Metaculus community predictions (MiniBench: "Will the community
prediction be higher than 31.00% on 2026-01-30 for the Metaculus question '...'?").

We read the linked question's recency-weighted community prediction (CP) history and
treat the CP as a random walk on the log-odds scale with a shrunken recent drift.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from nexora.aggregation import logit
from nexora.quant.stats import normal_cdf

logger = logging.getLogger(__name__)

_CP_RE = re.compile(
    r"community\s+prediction\s+be\s+(?P<direction>higher|lower)\s+than\s+(?P<threshold>\d+(?:\.\d+)?)\s*%\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_QUESTION_URL_RE = re.compile(r"metaculus\.com/questions/(\d+)", re.IGNORECASE)
MIN_DAILY_LOGIT_VOL = 0.03


@dataclass(frozen=True)
class CommunitySpec:
    threshold: float  # fraction
    target_date: date
    direction: str  # "higher" or "lower"
    linked_post_id: int | None


def detect(texts: dict[str, str], own_post_id: int | None) -> CommunitySpec | None:
    match = None
    for key in ("question_text", "post_title", "resolution_criteria"):
        match = _CP_RE.search(texts.get(key) or "")
        if match:
            break
    if not match:
        return None
    joined = "\n".join(v for v in texts.values() if v)
    linked = None
    for raw in _QUESTION_URL_RE.findall(joined):
        if own_post_id is None or int(raw) != own_post_id:
            linked = int(raw)
            break
    return CommunitySpec(
        threshold=float(match.group("threshold")) / 100.0,
        target_date=date.fromisoformat(match.group("date")),
        direction=match.group("direction").lower(),
        linked_post_id=linked,
    )


def daily_series_from_history(history: list[dict], now: datetime, days: int = 30) -> list[float]:
    """Sample the CP once per day (at the current time of day) over the last `days` days."""
    points: list[float] = []
    for back in range(days, -1, -1):
        ts = (now - timedelta(days=back)).timestamp()
        value = None
        for item in history:
            start, end = item.get("start_time"), item.get("end_time")
            if start is not None and start <= ts and (end is None or ts <= end):
                centers = item.get("centers") or []
                if centers:
                    value = float(centers[-1] if len(centers) > 1 else centers[0])
        if value is not None:
            points.append(value)
    return points


@dataclass
class CommunityForecast:
    probability: float
    summary: str


def forecast(spec: CommunitySpec, current_cp: float, history: list[dict] | None, now: datetime | None = None) -> CommunityForecast:
    now = now or datetime.now(timezone.utc)
    target_moment = datetime.combine(spec.target_date, time(12, 0), tzinfo=timezone.utc)
    horizon_days = max((target_moment - now).total_seconds() / 86400.0, 0.0)

    series = daily_series_from_history(history or [], now) if history else []
    if len(series) >= 5:
        z = np.array([logit(v) for v in series])
        diffs = np.diff(z)
        vol = max(float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0, MIN_DAILY_LOGIT_VOL)
        recent = diffs[-14:]
        drift = 0.5 * float(np.mean(recent)) if recent.size else 0.0
        history_note = f"{len(series)} daily CP samples; daily log-odds volatility {vol:.3f}; recent drift {drift:+.4f}/day (shrunk 50%)"
    else:
        vol, drift = 0.06, 0.0
        history_note = "no usable CP history; assumed daily log-odds volatility 0.06 and no drift"

    z_now, z_thr = logit(current_cp), logit(spec.threshold)
    if horizon_days <= 0:
        p_higher = 1.0 if current_cp > spec.threshold else 0.0
    else:
        p_higher = normal_cdf((z_now + drift * horizon_days - z_thr) / (vol * math.sqrt(horizon_days)))
    probability = p_higher if spec.direction == "higher" else 1.0 - p_higher
    probability = min(max(probability, 0.03), 0.97)
    summary = (
        f"Linked question CP now {current_cp:.2%} vs threshold {spec.threshold:.2%} "
        f"({spec.direction} needed) with {horizon_days:.1f} days until {spec.target_date.isoformat()}. "
        f"Model: random walk on log-odds, {history_note}. Probability of resolving Yes: {probability:.1%}.\n"
        "Note: CPs are sticky; for 'will X happen by <deadline>' questions they tend to drift down as time passes without news."
    )
    return CommunityForecast(probability=probability, summary=summary)
