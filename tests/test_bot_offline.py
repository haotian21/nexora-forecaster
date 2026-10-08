"""
End-to-end run of the bot with every external service faked: no network, no keys.
Checks research -> ensemble forecasts -> aggregation -> comment-first publishing.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

import nexora.bot as bot_module
import nexora.research as research_module
from nexora.bot import NexoraBot
from nexora.config import PROFILES, Settings
from nexora.models import ModelRegistry
from nexora.prompts import NUMERIC_PERCENTILES
from nexora.quant import QuantResult

NOW = datetime.now(timezone.utc)


class FakeClient:
    def __init__(self, fail_numeric_post: bool = False):
        self.calls: list[tuple] = []
        self.comment_flags: list[tuple[bool, bool]] = []
        self.fail_numeric_post = fail_numeric_post

    def post_question_comment(self, post_id, text, is_private=True, included_forecast=True):
        self.calls.append(("comment", post_id, text))
        self.comment_flags.append((is_private, included_forecast))

    def post_binary_question_prediction(self, question_id, p):
        assert 0.001 <= p <= 0.999
        self.calls.append(("binary", question_id, p))

    def post_multiple_choice_question_prediction(self, question_id, options):
        self.calls.append(("mc", question_id, options))

    def post_numeric_question_prediction(self, question_id, cdf):
        if self.fail_numeric_post:
            raise RuntimeError("network down")
        assert len(cdf) in (201, 21)
        assert all(0 <= x <= 1 for x in cdf)
        assert all(a <= b for a, b in zip(cdf, cdf[1:]))
        self.calls.append(("numeric", question_id, cdf))

    def get_question_by_post_id(self, post_id):
        raise AssertionError("not expected in these tests")

    async def get_questions_matching_filter(self, api_filter, *args, **kwargs):
        assert api_filter.allowed_statuses == ["resolved"]
        return [
            BinaryQuestion(
                question_text="Will the thing happen by the end of October 2025?",
                id_of_post=77, id_of_question=7077, page_url="https://www.metaculus.com/questions/77/",
                resolution_string="no", actual_resolution_time=NOW - timedelta(days=330),
                community_prediction_at_access_time=0.08,
            ),
            BinaryQuestion(
                question_text="Unrelated question about volcanoes erupting?",
                id_of_post=78, id_of_question=7078, resolution_string="yes",
            ),
        ]


def _common(post_id: int) -> dict:
    return dict(
        id_of_post=post_id,
        id_of_question=post_id + 1000,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        resolution_criteria="Resolves according to the official announcement.",
        fine_print="",
        background_info="Some background.",
        close_time=NOW + timedelta(hours=1),
        scheduled_resolution_time=NOW + timedelta(days=20),
        already_forecasted=False,
    )


def make_questions():
    binary = BinaryQuestion(question_text="Will the thing happen by the end of October 2026?", **_common(1))
    mc = MultipleChoiceQuestion(
        question_text="Will the interest in 'widgets' change between 2026-10-01 and 2026-10-09 according to Google Trends?",
        options=["Increases", "Doesn't change", "Decreases"],
        **_common(2),
    )
    numeric_open = NumericQuestion(
        question_text="What will be the value of 'Market Yield on 5-Year Treasuries' on 2026-10-09?",
        upper_bound=5.0,
        lower_bound=3.0,
        open_upper_bound=True,
        open_lower_bound=True,
        unit_of_measure="Percent",
        **_common(3),
    )
    numeric_closed = NumericQuestion(
        question_text="How many widgets (0-100) will be sold?",
        upper_bound=100.0,
        lower_bound=0.0,
        open_upper_bound=False,
        open_lower_bound=False,
        unit_of_measure="widgets",
        **_common(4),
    )
    date_q = DateQuestion(
        question_text="When will the widget launch?",
        upper_bound=NOW + timedelta(days=400),
        lower_bound=NOW - timedelta(days=10),
        open_upper_bound=True,
        open_lower_bound=False,
        **_common(5),
    )
    return [binary, mc, numeric_open, numeric_closed, date_q]


def fake_answer(prompt: str, model: str) -> str:
    if "FINAL PROBABILITIES" in prompt:
        return "analysis...\nFINAL PROBABILITIES\nIncreases: 30%\nDoesn't change: 20%\nDecreases: 50%"
    if "YYYY-MM-DD" in prompt:
        base = NOW + timedelta(days=30)
        return "\n".join(
            f"Percentile {p}: {(base + timedelta(days=3 * i)).date().isoformat()}" for i, p in enumerate(NUMERIC_PERCENTILES)
        )
    if "Percentile 5: X" in prompt:
        if "Treasuries" in prompt:
            vals = [3.6, 3.7, 3.8, 3.85, 3.9, 3.93, 3.96, 4.0, 4.05, 4.15, 4.25]
        else:
            vals = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]
        return "reasoning\n" + "\n".join(f"Percentile {p}: {v}" for p, v in zip(NUMERIC_PERCENTILES, vals))
    probability = {"claude": 30, "openai": 40, "gemini": 35}.get(model.split("/")[0].replace("anthropic", "claude").replace("google", "gemini"), 33)
    return f"Base rate reasoning...\nProbability: {probability}%"


@pytest.fixture
def patched(monkeypatch):
    async def fake_call_slot(plan, prompt, system_prompt=None):
        if plan.slot.name == "helper":
            return '{"news_query": "widgets", "market_queries": ["widgets"]}', plan.primary
        if plan.slot.name == "web":
            return "Web research: nothing decisive.", plan.primary
        return fake_answer(prompt, plan.primary), plan.primary

    async def no_quant(question, metaculus_client=None, timeout=25):
        return None

    monkeypatch.setattr(bot_module, "call_slot", fake_call_slot)
    monkeypatch.setattr(research_module, "call_slot", fake_call_slot)
    monkeypatch.setattr(research_module, "run_quant", no_quant)
    monkeypatch.setattr(research_module, "asknews_configured", lambda: False)
    monkeypatch.setattr(research_module.markets_mod, "search_all", lambda queries, timeout=20: [])
    return fake_call_slot


def make_bot(client, publish=True, profile="standard"):
    settings = Settings.from_env(publish=publish, profile_name=profile)
    return NexoraBot(settings, ModelRegistry(None), metaculus_client=client)


async def test_full_run_publishes_comment_first(patched):
    client = FakeClient()
    bot = make_bot(client)
    questions = make_questions()
    reports = await bot.forecast_questions(questions, return_exceptions=True)
    errors = [r for r in reports if isinstance(r, BaseException)]
    assert not errors, errors
    assert len(reports) == len(questions)

    # every question: exactly one comment, posted before its forecast
    kinds = [c[0] for c in client.calls]
    assert kinds.count("comment") == len(questions)
    for q in questions:
        comment_index = next(i for i, c in enumerate(client.calls) if c[0] == "comment" and c[1] == q.id_of_post)
        forecast_index = next(i for i, c in enumerate(client.calls) if c[0] != "comment" and c[1] == q.id_of_question)
        assert comment_index < forecast_index, client.calls
    # private, and no forecast attached: none exists yet when the comment goes first
    assert client.comment_flags == [(True, False)] * len(questions)

    binary = next(c for c in client.calls if c[0] == "binary")
    # 5 forecasters: two claude (30%), two openai (40%), one gemini (35%) -> trimmed log-odds mean ~35%
    assert 0.32 < binary[2] < 0.38
    mc = next(c for c in client.calls if c[0] == "mc")[2]
    assert sum(mc.values()) == pytest.approx(1.0)
    assert min(mc.values()) >= 0.01
    numerics = [c for c in client.calls if c[0] == "numeric"]
    assert len(numerics) == 3
    closed = next(c for c in numerics if c[1] == 1004)[2]
    assert closed[0] == pytest.approx(0.0) and closed[-1] == pytest.approx(1.0)

    comment = next(c for c in client.calls if c[0] == "comment" and c[1] == 1)[2]
    assert "Forecaster model" in comment
    assert "gemini" in comment.lower()
    # similar resolved question found (relevant one only) and shown as base-rate evidence
    assert "Similar RESOLVED Metaculus questions" in comment
    assert "end of October 2025? -> resolved NO" in comment
    assert "volcanoes" not in comment


async def test_one_failing_slot_does_not_lose_the_question(patched, monkeypatch):
    calls = {"n": 0}

    async def flaky(plan, prompt, system_prompt=None):
        if plan.slot.name.startswith("openai"):
            raise RuntimeError("provider outage")
        return await patched(plan, prompt, system_prompt)

    monkeypatch.setattr(bot_module, "call_slot", flaky)
    client = FakeClient()
    bot = make_bot(client)
    report = (await bot.forecast_questions(make_questions()[:1], return_exceptions=True))[0]
    assert not isinstance(report, BaseException), report
    assert "provider outage" in " ".join(report.errors)


async def test_failed_forecast_post_after_comment_raises(patched):
    client = FakeClient(fail_numeric_post=True)
    bot = make_bot(client)
    reports = await bot.forecast_questions(make_questions()[2:3], return_exceptions=True)
    assert isinstance(reports[0], BaseException)
    assert [c[0] for c in client.calls] == ["comment"]  # retried next run (question still unforecasted)


async def test_dry_run_posts_nothing(patched):
    client = FakeClient()
    bot = make_bot(client, publish=False, profile="lean")
    reports = await bot.forecast_questions(make_questions(), return_exceptions=True)
    assert all(not isinstance(r, BaseException) for r in reports)
    assert client.calls == []


async def test_forecasting_disabled_stops_publishing(patched):
    refused = '403 Forbidden: {"detail":"API forecasting is not enabled","code":"api_forecasting_not_enabled"}'

    class RefusingClient(FakeClient):
        def post_binary_question_prediction(self, question_id, p):
            raise RuntimeError(refused)

        def post_multiple_choice_question_prediction(self, question_id, options):
            raise RuntimeError(refused)

        def post_numeric_question_prediction(self, question_id, cdf):
            raise RuntimeError(refused)

    client = RefusingClient()
    bot = make_bot(client)
    first = await bot.forecast_questions(make_questions(), return_exceptions=True)
    assert all(isinstance(r, BaseException) for r in first)
    comments = [c for c in client.calls if c[0] == "comment"]
    assert 1 <= len(comments) <= len(first)  # only questions already publishing when the refusal came
    second = await bot.forecast_questions(make_questions(), return_exceptions=True)
    assert all(isinstance(r, BaseException) and "Not publishing" in str(r) for r in second)
    assert [c for c in client.calls if c[0] == "comment"] == comments  # nothing more posted


async def test_quant_blend_for_numeric(patched, monkeypatch):
    async def fred_quant(question, metaculus_client=None, timeout=25):
        if "Treasuries" not in question.question_text:
            return None
        pts = [(p / 100, 3.5 + p / 100) for p in range(1, 100, 2)]
        return QuantResult(kind="numeric", source="FRED:DGS5", summary="model says ~4.0", points=pts)

    monkeypatch.setattr(research_module, "run_quant", fred_quant)
    client = FakeClient()
    bot = make_bot(client)
    reports = await bot.forecast_questions(make_questions()[2:3], return_exceptions=True)
    report = reports[0]
    assert not isinstance(report, BaseException), report
    assert "Statistical model (FRED:DGS5)" in report.explanation
    median = report.prediction.get_percentiles_at_target_heights([0.5])[0].value
    assert 3.85 < median < 4.1


async def test_units_mistake_is_rejected(patched, monkeypatch):
    async def wrong_units(plan, prompt, system_prompt=None):
        if plan.slot.name in ("helper", "web"):
            return await patched(plan, prompt, system_prompt)
        vals = [360, 370, 380, 385, 390, 393, 396, 400, 405, 415, 425]  # basis points instead of percent
        return "\n".join(f"Percentile {p}: {v}" for p, v in zip(NUMERIC_PERCENTILES, vals)), plan.primary

    monkeypatch.setattr(bot_module, "call_slot", wrong_units)
    client = FakeClient()
    bot = make_bot(client)
    reports = await bot.forecast_questions(make_questions()[2:3], return_exceptions=True)
    assert isinstance(reports[0], BaseException)
    assert "units" in str(reports[0]).lower() or "far outside" in str(reports[0]).lower()
    assert client.calls == []


async def test_deadline_defers_questions(patched):
    import time

    client = FakeClient()
    settings = Settings.from_env(publish=True, profile_name="lean")
    bot = NexoraBot(settings, ModelRegistry(None), deadline=time.time() - 1, metaculus_client=client)
    reports = await bot.forecast_questions(make_questions()[:2], return_exceptions=True)
    assert all(isinstance(r, BaseException) and "Run time budget used up" in str(r) for r in reports)
    assert client.calls == []


def test_prompts_contain_key_instructions():
    from nexora import prompts

    q = make_questions()
    text = prompts.binary_prompt(q[0], "research", None)
    assert "Probability: NN%" in text and "Bayesian" not in text
    mc_text = prompts.multiple_choice_prompt(q[1], "research", None)
    assert "GOOGLE TRENDS" in mc_text and "FINAL PROBABILITIES" in mc_text
    num_text = prompts.numeric_prompt(q[2], "research", "MODEL SUMMARY")
    assert "STATISTICAL MODEL" in num_text and "MODEL SUMMARY" in num_text
    assert re.search(r"Percentile 95: X", num_text)
