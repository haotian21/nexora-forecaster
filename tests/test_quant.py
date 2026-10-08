import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from nexora.quant import community, fred, stocks
from nexora.quant.stats import business_days_after, normal_cdf, percentile_points


def business_dates(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_business_days_after():
    fri = date(2026, 9, 25)
    assert business_days_after(fri, date(2026, 9, 28)) == 1  # Mon
    assert business_days_after(fri, date(2026, 9, 27)) == 0  # Sun
    assert business_days_after(fri, date(2026, 10, 2)) == 5


def test_fred_detect_variants():
    spec = fred.detect(
        {"question_text": "What will be the value of 'Market Yield ... 5-Year' on 2026-04-08?",
         "post_title": "What will the value of FRED series DGS5 be?"},
        None,
    )
    assert spec == fred.FredSpec("DGS5", date(2026, 4, 8))
    spec2 = fred.detect(
        {"question_text": "What will the 10y breakeven be on 2026-11-02?",
         "resolution_criteria": "Resolves to https://fred.stlouisfed.org/series/t10yie on that date"},
        None,
    )
    assert spec2.series_id == "T10YIE"
    assert fred.detect({"question_text": "Will FRED series data be revised?"}, None) is None


def test_fred_parse_rows_handles_missing_values():
    rows = [("2026-01-02", "4.1"), ("2026-01-05", "."), ("2026-01-06", ""), ("bad", "1"), ("2026-01-07", "4.2")]
    assert fred.parse_rows(rows) == [(date(2026, 1, 2), 4.1), (date(2026, 1, 7), 4.2)]


def _random_walk(n=900, sigma=0.05, start=4.0, seed=3):
    rng = np.random.default_rng(seed)
    dates = business_dates(date(2023, 1, 2), n)
    values = start + np.cumsum(rng.standard_t(df=5, size=n) * sigma / math.sqrt(5 / 3))
    return list(zip(dates, values.tolist()))


def test_fred_forecast_shape_and_width():
    obs = _random_walk()
    last_date = obs[-1][0]
    target = last_date + timedelta(days=14)  # 10 business days
    result = fred.forecast(fred.FredSpec("TEST", target), obs, today=last_date)
    values = [v for _, v in result.points]
    assert all(a < b for a, b in zip(values, values[1:]))
    median = dict(result.points)[0.5]
    assert median == pytest.approx(result.last_value, abs=0.05)
    width_90 = dict(result.points)[0.95] - dict(result.points)[0.05]
    # levels > 0 so log-changes are used; width should be ~ 2*1.645*sigma*sqrt(10) (within generous tolerance)
    expected = 2 * 1.645 * 0.05 * math.sqrt(10) * 1.1
    assert 0.6 * expected < width_90 < 1.6 * expected
    assert result.horizon_steps == 10
    assert "DGS5" not in result.summary and "TEST" in result.summary


def test_fred_forecast_past_target_is_narrow():
    obs = _random_walk()
    target = obs[-5][0]
    result = fred.forecast(fred.FredSpec("TEST", target), obs, today=obs[-1][0])
    lookup = dict(result.points)
    assert lookup[0.5] == pytest.approx(obs[-5][1], rel=1e-2)
    assert lookup[0.95] - lookup[0.05] < 0.05


def test_fred_interval_coverage_is_calibrated():
    """Backtest the simulation on a synthetic series with volatility regimes: the central
    90% interval should contain the realised value roughly 90% of the time."""
    rng = np.random.default_rng(11)
    n = 1600
    vol = np.where((np.arange(n) // 200) % 2 == 0, 0.03, 0.08)
    series = 3.0 + np.cumsum(vol * rng.standard_normal(n))
    series = np.abs(series) + 0.5
    dates = business_dates(date(2020, 1, 1), n)
    hits, trials = 0, 0
    for end in range(800, n - 10, 15):
        obs = list(zip(dates[:end], series[:end].tolist()))
        target_idx = end - 1 + 5
        result = fred.forecast(fred.FredSpec("SYN", dates[target_idx]), obs, today=dates[end - 1])
        lookup = dict(result.points)
        realised = series[target_idx]
        hits += lookup[0.05] <= realised <= lookup[0.95]
        trials += 1
    coverage = hits / trials
    assert 0.8 <= coverage <= 0.99, coverage


def test_stock_detect_template():
    spec = stocks.detect(
        {"question_text": "Will MCO's market close price on 2026-03-13 be higher than its market close price on 2026-03-04?"}
    )
    assert spec == stocks.DirectionSpec("MCO", date(2026, 3, 4), date(2026, 3, 13))
    brk = stocks.detect(
        {"question_text": "Will BRK.B's market close price on 2026-03-13 be higher than its market close price on 2026-03-04?"}
    )
    assert brk.yahoo_symbol == "BRK-B"
    assert stocks.detect({"question_text": "Will Apple close higher?"}) is None


def _history(prices: list[float], end: date) -> stocks.PriceHistory:
    dates = business_dates(end - timedelta(days=int(len(prices) * 1.5) + 10), len(prices))
    shift = end - dates[-1]
    dates = [d + shift for d in dates]
    return stocks.PriceHistory(closes=list(zip(dates, prices)), dividends=[])


def test_stock_forecast_unknown_reference_is_near_half():
    rng = np.random.default_rng(5)
    prices = (100 * np.exp(np.cumsum(0.015 * rng.standard_normal(400)))).tolist()
    hist = _history(prices, date(2026, 9, 25))
    spec = stocks.DirectionSpec("XYZ", date(2026, 9, 30), date(2026, 10, 9))
    out = stocks.forecast(spec, hist)
    assert 0.45 < out.probability < 0.55


def test_stock_forecast_known_reference_moves_probability():
    rng = np.random.default_rng(6)
    prices = (100 * np.exp(np.cumsum(0.01 * rng.standard_normal(400)))).tolist()
    end = date(2026, 9, 25)
    hist = _history(prices, end)
    closes = hist.closes
    ref_date = closes[-3][0]
    # make the latest close 4% above the reference close, two trading days before the target
    closes[-1] = (closes[-1][0], closes[-3][1] * 1.04)
    spec = stocks.DirectionSpec("XYZ", ref_date, end + timedelta(days=3))  # Monday after
    out = stocks.forecast(spec, hist)
    assert out.probability > 0.9


def test_parse_chart_json():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD", "gmtoffset": -14400},
                    "timestamp": [1758807000, 1758893400],
                    "indicators": {"quote": [{"close": [100.5, None]}]},
                    "events": {"dividends": {"1758807000": {"amount": 0.5, "date": 1758807000}}},
                }
            ],
            "error": None,
        }
    }
    hist = stocks.parse_chart_json(payload)
    assert len(hist.closes) == 1 and hist.closes[0][1] == 100.5
    assert hist.dividends[0][1] == 0.5
    with pytest.raises(ValueError):
        stocks.parse_chart_json({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_parse_nasdaq_json():
    historical = {
        "data": {
            "tradesTable": {
                "rows": [
                    {"date": "10/07/2026", "close": "$1,336.67"},
                    {"date": "10/06/2026", "close": "777.22"},  # ETFs come without "$"
                    {"date": "10/03/2026", "close": "N/A"},
                ]
            }
        }
    }
    dividends = {
        "data": {
            "dividends": {
                "rows": [
                    {"exOrEffDate": "08/10/2026", "amount": "$0.27"},
                    {"exOrEffDate": "N/A", "amount": "$0.27"},
                    {"exOrEffDate": "05/11/2026", "amount": "$0.26"},
                ]
            }
        }
    }
    hist = stocks.parse_nasdaq_json(historical, dividends)
    assert hist.closes == [(date(2026, 10, 6), 777.22), (date(2026, 10, 7), 1336.67)]
    assert hist.dividends == [(date(2026, 5, 11), 0.26), (date(2026, 8, 10), 0.27)]
    assert hist.source == "Nasdaq" and hist.currency == "USD"
    assert stocks.parse_nasdaq_json(historical).dividends == []
    with pytest.raises(ValueError):
        stocks.parse_nasdaq_json({"data": None, "status": {"rCode": 400}})


def test_fetch_history_falls_back_to_nasdaq(monkeypatch):
    nasdaq = stocks.PriceHistory(closes=[(date(2026, 10, 7), 10.0)], dividends=[], source="Nasdaq")

    def yahoo_429(ticker, timeout):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(stocks, "fetch_yahoo", yahoo_429)
    monkeypatch.setattr(stocks, "fetch_nasdaq", lambda ticker, timeout: nasdaq)
    assert stocks.fetch_history("BRK.B") is nasdaq

    def nasdaq_down(ticker, timeout):
        raise ValueError("no rows")

    monkeypatch.setattr(stocks, "fetch_nasdaq", nasdaq_down)
    with pytest.raises(RuntimeError, match="Yahoo: RuntimeError: 429.*Nasdaq: ValueError: no rows"):
        stocks.fetch_history("BRK.B")


def test_community_detect_and_forecast():
    texts = {
        "question_text": "Will the community prediction be higher than 31.00% on 2026-01-30 for the Metaculus question "
        "'Will there be a ceasefire in the Sudanese Civil War during 2026?'?",
        "resolution_criteria": "See https://www.metaculus.com/questions/12345/ for the linked question.",
    }
    spec = community.detect(texts, own_post_id=999)
    assert spec.threshold == pytest.approx(0.31)
    assert spec.linked_post_id == 12345
    now = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)
    flat = community.forecast(spec, 0.31, None, now)
    assert 0.3 < flat.probability < 0.7
    above = community.forecast(spec, 0.40, None, now)
    assert above.probability > 0.8
    # history with a steady downward drift pushes the probability down
    history = []
    for i in range(40):
        t0 = (now - timedelta(days=40 - i)).timestamp()
        history.append({"start_time": t0, "end_time": t0 + 86400, "centers": [0.40 - 0.002 * i]})
    drifting = community.forecast(spec, 0.32, history, now)
    assert drifting.probability < community.forecast(spec, 0.32, None, now).probability


def test_normal_cdf_and_percentile_points():
    assert normal_cdf(0) == pytest.approx(0.5)
    assert normal_cdf(1.6448536) == pytest.approx(0.95, abs=1e-6)
    pts = percentile_points(np.array([1.0] * 100), [0.1, 0.5, 0.9])
    assert pts[0][1] < pts[1][1] < pts[2][1]
