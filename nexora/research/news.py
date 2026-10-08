"""AskNews news search (free for FutureEval bot makers: ~1k calls/month, 4k per tournament).

A latest-news search (past 48 hours) costs 1 call, an archive search 5, so profiles can
skip the archive (`Profile.asknews_archive`) to stay inside the free quota.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from asknews_sdk import AsyncAskNewsSDK
from forecasting_tools import AskNewsSearcher

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 12.0  # free tier: roughly one request per 10 seconds, across questions too

_lock: asyncio.Lock | None = None
_last_call = 0.0


def asknews_configured() -> bool:
    has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID", "").strip() and os.getenv("ASKNEWS_SECRET", "").strip())
    return has_oauth or bool(os.getenv("ASKNEWS_API_KEY", "").strip())


async def asknews_research(query: str, archive: bool = True) -> str:
    """Latest news, plus the 160-day archive if `archive`. Calls are serialised and spaced
    because of the free tier's rate limit (the library only sleeps between its own two calls)."""
    global _lock, _last_call
    if not asknews_configured():
        return ""
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        wait = _last_call + RATE_LIMIT_SECONDS - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            searcher = AskNewsSearcher()
            if archive:
                return await searcher.get_formatted_news_async(query[:400])
            return await _latest_news(searcher, query[:400])
        finally:
            _last_call = time.monotonic()


async def _latest_news(searcher: AskNewsSearcher, query: str) -> str:
    async with AsyncAskNewsSDK(
        client_id=searcher.client_id,
        client_secret=searcher.client_secret,
        api_key=searcher.api_key,
        scopes={"news"},
    ) as ask:
        response = await ask.news.search_news(query=query, n_articles=8, return_type="both", strategy="latest news")
    articles = response.as_dicts
    header = "Here are the relevant news articles:\n\n"
    if not articles:
        return header + "No articles were found.\n\n"
    return header + searcher._format_articles(articles)  # same formatting as the archive path
