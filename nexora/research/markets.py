"""
Read-only prediction-market prices as research input (Manifold + Polymarket public APIs).

Nothing is traded: the bot only reads public prices, which top bots use as strong
evidence when a liquid market asks the same question.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)
USER_AGENT = "nexora-forecaster/0.1 (research bot; read-only)"


@dataclass
class Market:
    platform: str
    title: str
    probability: float
    volume: float | None
    close: str | None
    url: str
    traders: int | None = None


def search_manifold(query: str, limit: int = 6, timeout: int = 20) -> list[Market]:
    response = requests.get(
        "https://api.manifold.markets/v0/search-markets",
        params={"term": query, "limit": limit, "filter": "open", "contractType": "BINARY"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_manifold(response.json())


def parse_manifold(items: list[dict]) -> list[Market]:
    markets = []
    for item in items or []:
        if item.get("outcomeType") != "BINARY" or item.get("isResolved"):
            continue
        prob = item.get("probability")
        if prob is None:
            continue
        close_ms = item.get("closeTime")
        close = (
            datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc).date().isoformat()
            if isinstance(close_ms, (int, float))
            else None
        )
        markets.append(
            Market(
                platform="Manifold (play money)",
                title=str(item.get("question", "")).strip(),
                probability=float(prob),
                volume=float(item["volume"]) if item.get("volume") is not None else None,
                close=close,
                url=str(item.get("url", "")),
                traders=item.get("uniqueBettorCount"),
            )
        )
    return markets


def search_polymarket(query: str, limit: int = 6, timeout: int = 20) -> list[Market]:
    response = requests.get(
        "https://gamma-api.polymarket.com/public-search",
        params={"q": query, "limit_per_type": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_polymarket(response.json())


def _loads_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_polymarket(payload: dict) -> list[Market]:
    markets = []
    for event in (payload or {}).get("events") or []:
        for market in event.get("markets") or []:
            if market.get("closed") or market.get("active") is False:
                continue
            outcomes = [str(o).lower() for o in _loads_list(market.get("outcomes"))]
            prices = _loads_list(market.get("outcomePrices"))
            if "yes" not in outcomes or len(prices) != len(outcomes):
                continue
            try:
                probability = float(prices[outcomes.index("yes")])
            except (TypeError, ValueError):
                continue
            try:
                volume = float(market.get("volume")) if market.get("volume") is not None else None
            except (TypeError, ValueError):
                volume = None
            slug = event.get("slug") or market.get("slug") or ""
            markets.append(
                Market(
                    platform="Polymarket (real money)",
                    title=str(market.get("question") or event.get("title") or "").strip(),
                    probability=probability,
                    volume=volume,
                    close=(market.get("endDate") or "")[:10] or None,
                    url=f"https://polymarket.com/event/{slug}" if slug else "",
                )
            )
    return markets


def format_markets(markets: list[Market], max_items: int = 8) -> str:
    if not markets:
        return ""
    seen, lines = set(), []
    ranked = sorted(markets, key=lambda m: -(m.volume or 0))
    for m in ranked:
        key = (m.platform, m.title.lower())
        if key in seen or not m.title:
            continue
        seen.add(key)
        volume = f", volume {m.volume:,.0f}" if m.volume is not None else ""
        traders = f", {m.traders} traders" if m.traders else ""
        close = f", closes {m.close}" if m.close else ""
        lines.append(f"- [{m.platform}] {m.title} -> {m.probability:.0%}{volume}{traders}{close} ({m.url})")
        if len(lines) >= max_items:
            break
    return "\n".join(lines)


def search_all(queries: list[str], timeout: int = 20) -> list[Market]:
    found: list[Market] = []
    for query in [q for q in queries if q and q.strip()][:2]:
        for search in (search_polymarket, search_manifold):
            try:
                found.extend(search(query.strip(), timeout=timeout))
            except Exception as e:  # noqa: BLE001
                logger.info(f"Market search failed ({search.__name__}, '{query}'): {e}")
    return found
