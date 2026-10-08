"""
Statistical models for auto-generated data-series questions. They run before the LLMs,
their summary is shown to every forecaster, and their output is blended into the final
forecast. Any failure (no match, network error, odd data) simply returns None.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from forecasting_tools import BinaryQuestion, DateQuestion, MetaculusQuestion, NumericQuestion

from nexora.quant import community, fred, stocks

logger = logging.getLogger(__name__)


@dataclass
class QuantResult:
    kind: Literal["numeric", "binary"]
    source: str
    summary: str
    points: list[tuple[float, float]] | None = None
    probability: float | None = None
    notes: list[str] = field(default_factory=list)


def question_texts(question: MetaculusQuestion) -> dict[str, str]:
    api = question.api_json or {}
    post_title = api.get("title") if isinstance(api, dict) else None
    return {
        "question_text": question.question_text or "",
        "post_title": post_title or "",
        "resolution_criteria": question.resolution_criteria or "",
        "fine_print": question.fine_print or "",
        "background_info": question.background_info or "",
    }


async def run_quant(question: MetaculusQuestion, metaculus_client=None, timeout: int = 25) -> QuantResult | None:
    texts = question_texts(question)
    try:
        if isinstance(question, NumericQuestion) and not isinstance(question, DateQuestion):
            spec = fred.detect(texts, question.scheduled_resolution_time)
            if spec:
                observations = await asyncio.to_thread(fred.fetch_series, spec.series_id, timeout)
                result = fred.forecast(spec, observations)
                return QuantResult(kind="numeric", source=f"FRED:{spec.series_id}", summary=result.summary, points=result.points)
        if isinstance(question, BinaryQuestion):
            direction = stocks.detect(texts)
            if direction:
                history = await asyncio.to_thread(stocks.fetch_history, direction.ticker, timeout)
                result = stocks.forecast(direction, history)
                return QuantResult(
                    kind="binary",
                    source=f"{history.source}:{direction.ticker.upper()}",
                    summary=result.summary,
                    probability=result.probability,
                )
            cp_spec = community.detect(texts, question.id_of_post)
            if cp_spec and cp_spec.linked_post_id and metaculus_client is not None:
                linked = await asyncio.to_thread(metaculus_client.get_question_by_post_id, cp_spec.linked_post_id)
                current = getattr(linked, "community_prediction_at_access_time", None)
                if current is not None:
                    history = None
                    try:
                        history = linked.api_json["question"]["aggregations"]["recency_weighted"]["history"]
                    except (KeyError, TypeError):
                        history = None
                    result = community.forecast(cp_spec, float(current), history, datetime.now(timezone.utc))
                    return QuantResult(
                        kind="binary",
                        source=f"MetaculusCP:{cp_spec.linked_post_id}",
                        summary=result.summary,
                        probability=result.probability,
                    )
    except Exception as e:  # noqa: BLE001 - quant is optional, never fatal
        logger.warning(f"Quant model skipped for {question.page_url}: {e.__class__.__name__}: {e}")
    return None
