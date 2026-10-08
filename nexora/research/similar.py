"""
Similar RESOLVED Metaculus questions as base-rate evidence. In the Fall 2025 bot survey,
34% of prize winners looked up similar past questions versus 0% of non-winners.
"""

from __future__ import annotations

import logging
import re

from forecasting_tools import BinaryQuestion, MetaculusQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']{2,}")
_STOP = {
    "will", "the", "before", "after", "than", "what", "which", "when", "does", "any", "there", "and", "for",
    "with", "from", "between", "least", "more", "less", "end", "during", "year", "next", "this", "that", "into",
    "by", "be", "of", "in", "on", "at", "to", "an", "or", "its", "their", "his", "her", "how", "many", "much",
}


def keywords(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP}


def pick_similar(question: MetaculusQuestion, candidates: list[MetaculusQuestion], limit: int = 5) -> list[BinaryQuestion]:
    target = keywords(question.question_text)
    scored = []
    for c in candidates:
        if not isinstance(c, BinaryQuestion) or c.id_of_question == question.id_of_question:
            continue
        if not isinstance(c.binary_resolution, bool):
            continue
        overlap = len(target & keywords(c.question_text))
        if overlap >= max(2, len(target) // 3):
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


def format_similar(questions: list[BinaryQuestion]) -> str:
    lines = []
    for q in questions:
        outcome = "YES" if q.binary_resolution else "NO"
        cp = q.community_prediction_at_access_time
        cp_text = f", final community prediction {cp:.0%}" if isinstance(cp, (int, float)) else ""
        when = q.actual_resolution_time.date().isoformat() if q.actual_resolution_time else "?"
        lines.append(f"- {q.question_text} -> resolved {outcome} ({when}{cp_text}) {q.page_url or ''}")
    return "\n".join(lines)


async def similar_resolved(question: MetaculusQuestion, client, search: str) -> str:
    if client is None or not search.strip():
        return ""
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        group_question_mode="exclude",
        other_url_parameters={"search": search.strip()[:120]},
    )
    # forecasting-tools' client is synchronous under an async signature; it runs briefly on the loop.
    candidates = await client.get_questions_matching_filter(api_filter)
    return format_similar(pick_similar(question, candidates))
