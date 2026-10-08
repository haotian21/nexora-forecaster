"""
Metaculus only accepts API forecasts from dedicated bot accounts. A personal account's
token can read questions and post comments, but every forecast is refused (403,
`api_forecasting_not_enabled`), so a run would leave a comment on each question and no
forecast. Checking the account type up front stops that before anything is posted.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

PARTICIPATE_URL = "https://www.metaculus.com/futureeval/participate/"
FORECASTING_DISABLED_CODE = "api_forecasting_not_enabled"


def metaculus_account(token: str | None = None, timeout: int = 20) -> dict | None:
    """The token's /users/me/ record, or None if it could not be fetched (never fatal)."""
    token = (token or os.getenv("METACULUS_TOKEN", "")).strip()
    base_url = os.getenv("METACULUS_API_BASE_URL") or "https://www.metaculus.com/api"
    try:
        response = requests.get(f"{base_url}/users/me/", headers={"Authorization": f"Token {token}"}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # noqa: BLE001 - the check is a convenience, not a gate on network trouble
        logger.warning(f"Could not check the Metaculus account type: {e}")
        return None


def account_problem(account: dict | None) -> str | None:
    if account is not None and account.get("is_bot") is False:
        return (
            f"METACULUS_TOKEN belongs to the personal account '{account.get('username')}', and Metaculus only accepts "
            f"API forecasts from a bot account. Create your bot at {PARTICIPATE_URL} and use the bot's token."
        )
    return None
