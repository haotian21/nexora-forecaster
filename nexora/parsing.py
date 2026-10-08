"""
Deterministic parsers for the forecasters' final answers.

Regex first (free, exact), LLM-based `structure_output` only as a fallback. Parsed
values are validated so malformed answers raise instead of silently becoming forecasts.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone

NUMBER = r"[-+]?\d[\d,_ ]*(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?"

_BINARY_RE = re.compile(r"probability\s*(?:of\s*yes)?\s*[:=]\s*\**\s*(" + NUMBER + r")\s*%", re.IGNORECASE)
_PERCENTILE_RE = re.compile(
    r"percentile\s*(\d{1,2}(?:\.\d+)?)\s*[:=]\s*\**\s*([^\n]*)",
    re.IGNORECASE,
)


class ParseError(ValueError):
    pass


def _to_float(raw: str) -> float:
    cleaned = raw.replace(",", "").replace("_", "").replace(" ", "").replace("−", "-")
    value = float(cleaned)
    if math.isnan(value) or math.isinf(value):
        raise ParseError(f"Non-finite number: {raw}")
    return value


def parse_binary(text: str) -> float:
    matches = _BINARY_RE.findall(text or "")
    if not matches:
        raise ParseError("No 'Probability: NN%' line found")
    value = _to_float(matches[-1]) / 100.0
    if not 0.0 <= value <= 1.0:
        raise ParseError(f"Probability out of range: {value}")
    return value


def _normalise_label(label: str) -> str:
    label = unicodedata.normalize("NFKC", label).casefold().strip()
    label = re.sub(r"^[\s\-*•>#\d.)(]+(?=\D)", "", label)  # bullets / numbering
    label = re.sub(r"^option\s*[a-z0-9]?\s*[:.)-]\s*", "", label)
    label = label.strip(" \t*_`\"'“”‘’")
    label = re.sub(r"\s+", " ", label)
    return label


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Parses lines `<option>: NN%` from the final block of the answer."""
    text = text or ""
    markers = list(re.finditer(r"final\s+probabilities", text, re.IGNORECASE))
    block = text[markers[-1].end():] if markers else text
    wanted = {_normalise_label(o): o for o in options}
    found: dict[str, float] = {}
    for line in block.splitlines():
        if ":" not in line or "%" not in line:
            continue
        name_part, _, value_part = line.rpartition(":")
        number = re.search(NUMBER, value_part)
        if not number:
            continue
        key = _normalise_label(name_part)
        if key in wanted:
            found[wanted[key]] = _to_float(number.group(0)) / 100.0
    if set(found) != set(options):
        missing = [o for o in options if o not in found]
        raise ParseError(f"Could not find probabilities for options: {missing}")
    total = sum(found.values())
    if total <= 0 or any(v < 0 for v in found.values()):
        raise ParseError("Invalid multiple-choice probabilities")
    if not 0.8 <= total <= 1.2:
        raise ParseError(f"Probabilities sum to {total:.2f}, expected ~1")
    return {o: v / total for o, v in found.items()}


def parse_numeric_percentiles(text: str, required: list[int]) -> list[tuple[float, float]]:
    """Returns [(percentile_as_fraction, value)] for the required percentiles (last occurrence wins)."""
    values: dict[int, float] = {}
    for pct_raw, rest in _PERCENTILE_RE.findall(text or ""):
        pct = float(pct_raw)
        if pct != int(pct):
            continue
        number = re.search(NUMBER, rest)
        if not number:
            continue
        try:
            values[int(pct)] = _to_float(number.group(0).strip())
        except ValueError:
            continue
    missing = [p for p in required if p not in values]
    if missing:
        raise ParseError(f"Missing percentiles {missing}")
    ordered = [(p / 100.0, values[p]) for p in sorted(required)]
    return enforce_increasing(ordered)


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)Z?)?")


def parse_date_percentiles(text: str, required: list[int]) -> list[tuple[float, float]]:
    """Like parse_numeric_percentiles but values are dates; returns POSIX timestamps."""
    values: dict[int, float] = {}
    for pct_raw, rest in _PERCENTILE_RE.findall(text or ""):
        pct = float(pct_raw)
        if pct != int(pct):
            continue
        match = _DATE_RE.search(rest)
        if not match:
            continue
        date_part, time_part = match.group(1), match.group(2) or "00:00:00"
        if len(time_part) == 5:
            time_part += ":00"
        try:
            dt = datetime.fromisoformat(f"{date_part}T{time_part}").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        values[int(pct)] = dt.timestamp()
    missing = [p for p in required if p not in values]
    if missing:
        raise ParseError(f"Missing date percentiles {missing}")
    ordered = [(p / 100.0, values[p]) for p in sorted(required)]
    return enforce_increasing(ordered)


def enforce_increasing(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Values must rise with the percentile. Sort values if the model listed them out of order,
    then nudge exact ties apart so the CDF stays strictly increasing."""
    percentiles = [p for p, _ in points]
    values = sorted(v for _, v in points)
    span = (values[-1] - values[0]) or max(abs(values[0]), 1.0)
    nudge = span * 1e-6
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + nudge
    return list(zip(percentiles, values))
