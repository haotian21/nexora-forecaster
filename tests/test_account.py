"""The bot must refuse to post from a personal Metaculus account (forecasts would all be refused)."""

import requests

from nexora import account


def test_personal_account_is_a_setup_problem():
    problem = account.account_problem({"is_bot": False, "username": "someone"})
    assert problem and "personal account 'someone'" in problem and account.PARTICIPATE_URL in problem


def test_bot_account_or_unknown_is_fine():
    assert account.account_problem({"is_bot": True, "username": "my-bot"}) is None
    assert account.account_problem(None) is None  # network trouble must not block a run


def test_account_lookup_failure_returns_none(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(account.requests, "get", down)
    assert account.metaculus_account("token") is None
