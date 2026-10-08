"""
Stock-direction questions (MiniBench: "Will MCO's market close price on 2026-03-13 be
higher than its market close price on 2026-03-04?").

Short-horizon returns are close to a random walk, so the honest answer is usually near
50%, unless the price has already moved away from the reference close. We price that with
a lognormal model using a blend of current and long-run volatility, a small equity drift
and the expected ex-dividend drop if one falls inside the window.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from nexora.quant.stats import blended_volatility, business_days_after, normal_cdf

logger = logging.getLogger(__name__)

_DIRECTION_RE = re.compile(
    r"will\s+(?P<ticker>[A-Za-z0-9.\-^=]{1,12})['’]s\s+(?:market\s+)?close\s+price\s+on\s+(?P<later>\d{4}-\d{2}-\d{2})\s+"
    r"be\s+higher\s+than\s+its\s+(?:market\s+)?close\s+price\s+on\s+(?P<earlier>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
NASDAQ_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
DAILY_DRIFT = 0.0002  # ~5% a year; deliberately modest


@dataclass(frozen=True)
class DirectionSpec:
    ticker: str
    earlier: date
    later: date

    @property
    def yahoo_symbol(self) -> str:
        return self.ticker.upper().replace(".", "-")


def detect(texts: dict[str, str]) -> DirectionSpec | None:
    for key in ("question_text", "post_title", "resolution_criteria"):
        match = _DIRECTION_RE.search(texts.get(key) or "")
        if match:
            earlier = date.fromisoformat(match.group("earlier"))
            later = date.fromisoformat(match.group("later"))
            if later <= earlier:
                return None
            return DirectionSpec(match.group("ticker"), earlier, later)
    return None


@dataclass
class PriceHistory:
    closes: list[tuple[date, float]]
    dividends: list[tuple[date, float]]
    currency: str | None = None
    source: str = "Yahoo Finance"


def fetch_history(ticker: str, timeout: int = 25) -> PriceHistory:
    """Daily closes from Yahoo Finance, or from Nasdaq's public API when Yahoo refuses (it often answers 429)."""
    errors = []
    for name, fetch in (("Yahoo", fetch_yahoo), ("Nasdaq", fetch_nasdaq)):
        try:
            return fetch(ticker, timeout)
        except Exception as e:  # noqa: BLE001 - try the next source
            errors.append(f"{name}: {e.__class__.__name__}: {str(e)[:150]}")
    raise RuntimeError("No stock price source worked: " + "; ".join(errors))


def fetch_yahoo(ticker: str, timeout: int = 25) -> PriceHistory:
    symbol = ticker.upper().replace(".", "-")
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": "2y", "interval": "1d", "events": "div"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_chart_json(response.json())


def parse_chart_json(payload: dict) -> PriceHistory:
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise ValueError(f"No chart data: {(payload.get('chart') or {}).get('error')}")
    data = result[0]
    offset = int((data.get("meta") or {}).get("gmtoffset") or 0)
    timestamps = data.get("timestamp") or []
    closes_raw = ((data.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    closes: list[tuple[date, float]] = []
    for ts, close in zip(timestamps, closes_raw):
        if close is None or not math.isfinite(close) or close <= 0:
            continue
        closes.append((datetime.fromtimestamp(ts + offset, tz=timezone.utc).date(), float(close)))
    dividends = []
    for item in ((data.get("events") or {}).get("dividends") or {}).values():
        try:
            dividends.append(
                (datetime.fromtimestamp(int(item["date"]) + offset, tz=timezone.utc).date(), float(item["amount"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    closes.sort()
    dividends.sort()
    return PriceHistory(closes=closes, dividends=dividends, currency=(data.get("meta") or {}).get("currency"))


def fetch_nasdaq(ticker: str, timeout: int = 25) -> PriceHistory:
    symbol = ticker.upper()  # Nasdaq keeps share-class dots (BRK.B)
    today = datetime.now(timezone.utc).date()
    params = {"fromdate": (today - timedelta(days=740)).isoformat(), "todate": today.isoformat(), "limit": 9999}
    for asset_class in ("stocks", "etf"):  # ETFs such as SPY are "Symbol not exists" under stocks
        response = requests.get(
            f"https://api.nasdaq.com/api/quote/{symbol}/historical",
            params={**params, "assetclass": asset_class},
            headers=NASDAQ_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        historical = response.json()
        if ((historical.get("data") or {}).get("tradesTable") or {}).get("rows"):
            break
    else:
        raise ValueError(f"Nasdaq has no price history for {symbol}: {historical.get('status')}")
    dividends = None
    try:
        response = requests.get(
            f"https://api.nasdaq.com/api/quote/{symbol}/dividends",
            params={"assetclass": asset_class},
            headers=NASDAQ_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        dividends = response.json()
    except Exception as e:  # noqa: BLE001 - dividends only refine the drift
        logger.info(f"Nasdaq dividends unavailable for {symbol}: {e}")
    return parse_nasdaq_json(historical, dividends)


def _nasdaq_number(raw: object) -> float:
    return float(str(raw).replace("$", "").replace(",", "").strip())


def _nasdaq_date(raw: object) -> date:
    return datetime.strptime(str(raw).strip(), "%m/%d/%Y").date()


def parse_nasdaq_json(historical: dict, dividends: dict | None = None) -> PriceHistory:
    closes: list[tuple[date, float]] = []
    for row in ((historical.get("data") or {}).get("tradesTable") or {}).get("rows") or []:
        try:
            day, close = _nasdaq_date(row["date"]), _nasdaq_number(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(close) and close > 0:
            closes.append((day, close))
    if not closes:
        raise ValueError(f"No Nasdaq price rows: {historical.get('status')}")
    divs: list[tuple[date, float]] = []
    for row in (((dividends or {}).get("data") or {}).get("dividends") or {}).get("rows") or []:
        try:
            divs.append((_nasdaq_date(row["exOrEffDate"]), _nasdaq_number(row["amount"])))
        except (KeyError, TypeError, ValueError):  # e.g. "N/A" dates
            continue
    closes.sort()
    divs.sort()
    return PriceHistory(closes=closes, dividends=divs, currency="USD", source="Nasdaq")


def _expected_dividend_drop(history: PriceHistory, window_start: date, window_end: date, price: float) -> tuple[float, str]:
    """Log-drift adjustment if the next projected ex-dividend date falls in (start, end]."""
    divs = history.dividends
    if len(divs) < 2 or price <= 0:
        return 0.0, ""
    gaps = np.diff([d.toordinal() for d, _ in divs[-6:]])
    if gaps.size == 0:
        return 0.0, ""
    step = float(np.median(gaps))
    next_ex = divs[-1][0].toordinal() + step
    while next_ex <= window_start.toordinal():
        next_ex += step
    if window_start.toordinal() < next_ex <= window_end.toordinal():
        amount = divs[-1][1]
        if 0 < amount < price:
            projected = date.fromordinal(int(round(next_ex)))
            return math.log(1 - amount / price), f"projected ex-dividend ~{projected.isoformat()} (~{amount:g} per share)"
    return 0.0, ""


@dataclass
class DirectionForecast:
    probability: float
    summary: str


def forecast(spec: DirectionSpec, history: PriceHistory) -> DirectionForecast:
    closes = history.closes
    if len(closes) < 60:
        raise ValueError(f"Not enough price history for {spec.ticker}")
    prices = np.array([c for _, c in closes])
    returns = np.diff(np.log(prices))[-500:]
    sigma = blended_volatility(returns)
    last_date, last_price = closes[-1]

    reference_known = spec.earlier <= last_date
    if reference_known:
        on_or_before = [c for d, c in closes if d <= spec.earlier]
        if not on_or_before:
            raise ValueError("Reference date precedes the price history")
        reference_price = on_or_before[-1]
        steps = business_days_after(last_date, spec.later)
        mean = math.log(last_price / reference_price)
        window_start = last_date
    else:
        reference_price = None
        steps = business_days_after(spec.earlier, spec.later)
        mean = 0.0
        window_start = spec.earlier

    div_adj, div_note = _expected_dividend_drop(history, window_start, spec.later, last_price)
    mean += (DAILY_DRIFT - 0.5 * sigma**2) * steps + div_adj
    spread = sigma * math.sqrt(steps)
    if spread <= 0:
        probability = 1.0 if mean > 0 else 0.0
    else:
        probability = normal_cdf(mean / spread)

    ref_text = (
        f"reference close on {spec.earlier.isoformat()} = {reference_price:g}"
        if reference_price is not None
        else f"reference close on {spec.earlier.isoformat()} not yet known"
    )
    summary = (
        f"{spec.ticker.upper()} daily closes from {history.source}: last close {last_date.isoformat()} = {last_price:g} "
        f"{history.currency or ''}; {ref_text}. Remaining trading days until {spec.later.isoformat()}: {steps}. "
        f"Daily volatility (80% EWMA, 20% 2-year): {sigma * 100:.2f}%.{(' ' + div_note + '.') if div_note else ''}\n"
        f"Lognormal random-walk probability that the {spec.later.isoformat()} close is strictly higher: {probability:.1%}. "
        "Short-horizon stock direction is close to unpredictable; news rarely justifies moving far from this number."
    )
    return DirectionForecast(probability=probability, summary=summary)
