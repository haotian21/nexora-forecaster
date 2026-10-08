"""
Parallel, multi-source research. Breadth of sources correlated with tournament score in
published bot surveys, so each question gets (when configured): a statistical model for
data-series questions, AskNews articles, a search-grounded web report and read-only
prediction-market prices. Every source is optional and time-boxed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from forecasting_tools import MetaculusQuestion

from nexora import prompts
from nexora.config import Profile
from nexora.models import SlotPlan, call_slot
from nexora.quant import QuantResult, run_quant
from nexora.research import markets as markets_mod
from nexora.research.news import asknews_configured, asknews_research
from nexora.research.similar import similar_resolved

logger = logging.getLogger(__name__)

_STOP = {
    "will", "the", "be", "by", "before", "after", "in", "of", "a", "an", "on", "to", "at", "for", "than", "what",
    "which", "who", "when", "how", "many", "much", "is", "are", "does", "do", "any", "there", "and", "or", "its",
    "their", "his", "her", "from", "with", "as", "between", "least", "more", "less", "end", "during",
}


@dataclass
class ResearchBundle:
    text: str
    quant: QuantResult | None = None
    sources: list[str] = field(default_factory=list)


def heuristic_queries(question_text: str) -> tuple[str, list[str]]:
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}", " ", question_text)
    cleaned = re.sub(r"[^\w\s$%.-]", " ", cleaned)
    words = [w for w in cleaned.split() if w.lower() not in _STOP and len(w) > 1]
    return question_text[:200], [" ".join(words[:6])]


def parse_queries(raw: str, question_text: str) -> tuple[str, list[str]]:
    fallback_news, fallback_markets = heuristic_queries(question_text)
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return fallback_news, fallback_markets
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback_news, fallback_markets
    news = str(data.get("news_query") or "").strip() or fallback_news
    market_queries = [str(q).strip() for q in (data.get("market_queries") or []) if str(q).strip()]
    return news[:300], (market_queries[:2] or fallback_markets)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n[...truncated]"


async def _guard(coro, seconds: float, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Research source '{label}' failed: {e.__class__.__name__}: {str(e)[:200]}")
        return None


async def gather_research(
    question: MetaculusQuestion,
    profile: Profile,
    helper_plan: SlotPlan,
    web_plan: SlotPlan | None,
    metaculus_client=None,
) -> ResearchBundle:
    quant_task = asyncio.create_task(_guard(run_quant(question, metaculus_client), 90, "quant")) if profile.use_quant else None
    web_task = (
        asyncio.create_task(_guard(call_slot(web_plan, prompts.research_prompt(question)), 300, "web"))
        if web_plan is not None
        else None
    )

    news_query, market_queries = heuristic_queries(question.question_text)
    raw_queries = await _guard(call_slot(helper_plan, prompts.query_prompt(question)), 90, "queries")
    if raw_queries:
        news_query, market_queries = parse_queries(raw_queries[0], question.question_text)

    news_task = (
        asyncio.create_task(_guard(asknews_research(news_query), 240, "asknews"))
        if profile.use_asknews and asknews_configured()
        else None
    )
    markets_task = (
        asyncio.create_task(_guard(asyncio.to_thread(markets_mod.search_all, market_queries), 60, "markets"))
        if profile.use_markets
        else None
    )
    similar_task = (
        asyncio.create_task(
            _guard(similar_resolved(question, metaculus_client, market_queries[0] if market_queries else ""), 90, "similar")
        )
        if profile.use_similar and metaculus_client is not None
        else None
    )

    quant: QuantResult | None = await quant_task if quant_task else None
    web = await web_task if web_task else None
    news = await news_task if news_task else None
    found_markets = await markets_task if markets_task else None
    similar = await similar_task if similar_task else None

    budget = profile.max_research_chars
    sections: list[str] = []
    sources: list[str] = []
    if quant:
        sources.append(quant.source)
    if web:
        text, model = web
        sections.append(f"## Web research ({model})\n{_truncate(text, int(budget * 0.4))}")
        sources.append(model)
    if news:
        sections.append(f"## News articles (AskNews, query: '{news_query}')\n{_truncate(news, int(budget * 0.45))}")
        sources.append("asknews")
    if found_markets:
        formatted = markets_mod.format_markets(found_markets)
        if formatted:
            sections.append(
                "## Prediction markets found by keyword search (may be unrelated or differ in deadline/criteria)\n"
                + _truncate(formatted, int(budget * 0.15))
            )
            sources.append("markets")
    if similar:
        sections.append(
            "## Similar RESOLVED Metaculus questions (use for base rates; they are different questions)\n"
            + _truncate(similar, int(budget * 0.1))
        )
        sources.append("similar-resolved")
    if not sections:
        sections.append("No external research could be retrieved for this question; rely on base rates and general knowledge.")
    return ResearchBundle(text="\n\n".join(sections), quant=quant, sources=sources)
