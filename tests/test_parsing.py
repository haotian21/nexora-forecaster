from datetime import datetime, timezone

import pytest

from nexora.parsing import (
    ParseError,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_numeric_percentiles,
)
from nexora.prompts import NUMERIC_PERCENTILES


@pytest.mark.parametrize(
    "text,expected",
    [
        ("blah\nProbability: 35%", 0.35),
        ("**Probability: 7.5%**", 0.075),
        ("Probability: **62 %**", 0.62),
        ("Early guess Probability: 10%\n...\nFinal Probability: 12%", 0.12),
    ],
)
def test_parse_binary(text, expected):
    assert parse_binary(text) == pytest.approx(expected)


def test_parse_binary_errors():
    with pytest.raises(ParseError):
        parse_binary("I think it is likely.")
    with pytest.raises(ParseError):
        parse_binary("Probability: 135%")


def test_parse_multiple_choice_variants():
    options = ["Increases", "Doesn't change", "Decreases"]
    text = """analysis with Increases: maybe 50% earlier
FINAL PROBABILITIES
- **Increases**: 30%
- Doesn't change: 25%
- Decreases: 45%"""
    out = parse_multiple_choice(text, options)
    assert out == pytest.approx({"Increases": 0.30, "Doesn't change": 0.25, "Decreases": 0.45})


def test_parse_multiple_choice_normalises_and_errors():
    options = ["0", "1-2", "3 or more"]
    text = "FINAL PROBABILITIES\n0: 33%\n1-2: 33%\n3 or more: 33%"
    out = parse_multiple_choice(text, options)
    assert sum(out.values()) == pytest.approx(1.0)
    with pytest.raises(ParseError):
        parse_multiple_choice("FINAL PROBABILITIES\n0: 50%\n1-2: 50%", options)


def test_parse_numeric_percentiles_with_commas_and_order():
    lines = "\n".join(f"Percentile {p}: {1000 + p * 10:,}" for p in NUMERIC_PERCENTILES)
    points = parse_numeric_percentiles("reasoning...\n" + lines, NUMERIC_PERCENTILES)
    assert points[0] == (0.05, 1050.0)
    assert points[-1] == (0.95, 1950.0)
    # values given in the wrong order get sorted, ties get nudged apart
    messy = "\n".join(f"Percentile {p}: {v}" for p, v in zip(NUMERIC_PERCENTILES, [5, 4, 6, 6, 7, 8, 9, 10, 11, 12, 13]))
    fixed = parse_numeric_percentiles(messy, NUMERIC_PERCENTILES)
    values = [v for _, v in fixed]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_parse_numeric_negative_and_missing():
    lines = "\n".join(f"Percentile {p}: -{100 - p}.5" for p in NUMERIC_PERCENTILES)
    points = parse_numeric_percentiles(lines, NUMERIC_PERCENTILES)
    assert points[0][1] == pytest.approx(-95.5)
    with pytest.raises(ParseError):
        parse_numeric_percentiles("Percentile 5: 1\nPercentile 95: 3", NUMERIC_PERCENTILES)


def test_parse_date_percentiles():
    text = "\n".join(f"Percentile {p}: 2027-01-{10 + i:02d}" for i, p in enumerate(NUMERIC_PERCENTILES))
    points = parse_date_percentiles(text, NUMERIC_PERCENTILES)
    assert points[0][1] == datetime(2027, 1, 10, tzinfo=timezone.utc).timestamp()
    assert len(points) == len(NUMERIC_PERCENTILES)
