"""AskNews news search (free for FutureEval bot makers: ~1k calls/month, 4k per tournament)."""

from __future__ import annotations

import asyncio
import logging
import os

from forecasting_tools import AskNewsSearcher

logger = logging.getLogger(__name__)

_lock: asyncio.Lock | None = None


def asknews_configured() -> bool:
    has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID", "").strip() and os.getenv("ASKNEWS_SECRET", "").strip())
    return has_oauth or bool(os.getenv("ASKNEWS_API_KEY", "").strip())


async def asknews_research(query: str) -> str:
    """Latest + historical articles. Calls are serialised because the free tier allows
    roughly one request per 10 seconds (the searcher sleeps between its two calls)."""
    global _lock
    if not asknews_configured():
        return ""
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        return await AskNewsSearcher().get_formatted_news_async(query[:400])
