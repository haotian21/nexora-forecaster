"""AskNews quota handling: latest-only searches, archive via the library, spacing between calls."""

import time

import pytest

from nexora.config import PROFILES
from nexora.research import news


class _Response:
    as_dicts: list = []


@pytest.fixture
def searches(monkeypatch):
    """Fakes the AskNews SDK and the library's two-search path; returns the searches made."""
    monkeypatch.setenv("ASKNEWS_API_KEY", "test-key")
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
    monkeypatch.setattr(news, "_lock", None)
    monkeypatch.setattr(news, "_last_call", 0.0)
    monkeypatch.setattr(news, "RATE_LIMIT_SECONDS", 0.0)
    made: list[str] = []

    class FakeNews:
        async def search_news(self, **kwargs):
            made.append(kwargs["strategy"])
            return _Response()

    class FakeSDK:
        def __init__(self, **kwargs):
            self.news = FakeNews()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def library_both(self, query):
        made.append("latest news + news knowledge")
        return "both"

    monkeypatch.setattr(news, "AsyncAskNewsSDK", FakeSDK)
    monkeypatch.setattr(news.AskNewsSearcher, "get_formatted_news_async", library_both)
    return made


async def test_latest_only_costs_one_search(searches):
    text = await news.asknews_research("query", archive=False)
    assert searches == ["latest news"]
    assert "No articles were found" in text


async def test_archive_uses_both_library_searches(searches):
    assert await news.asknews_research("query") == "both"
    assert searches == ["latest news + news knowledge"]


async def test_calls_are_spaced_across_questions(searches, monkeypatch):
    monkeypatch.setattr(news, "RATE_LIMIT_SECONDS", 0.3)
    started = time.monotonic()
    await news.asknews_research("a", archive=False)
    await news.asknews_research("b", archive=False)
    assert time.monotonic() - started >= 0.29
    assert searches == ["latest news", "latest news"]


async def test_not_configured_returns_empty(monkeypatch):
    for name in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
        monkeypatch.delenv(name, raising=False)
    assert await news.asknews_research("query") == ""


def test_only_lean_skips_the_archive():
    # ~4 MiniBench questions a day at 1 call + ~3-5 seasonal at 6 calls stays near the 1k/month quota
    assert {name for name, profile in PROFILES.items() if not profile.asknews_archive} == {"lean"}
