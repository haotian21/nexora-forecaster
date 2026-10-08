"""
Pre-flight check: verifies keys, model availability and data sources without forecasting.

  uv run python scripts/check_setup.py

Costs at most a fraction of a cent (one tiny LLM call). Exit code 1 if a required piece is broken.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dotenv  # noqa: E402

dotenv.load_dotenv()

import requests  # noqa: E402
from forecasting_tools import MetaculusClient  # noqa: E402

from nexora.account import account_problem, metaculus_account  # noqa: E402
from nexora.config import PROFILES  # noqa: E402
from nexora.models import ModelRegistry, SlotPlan, call_slot, openrouter_key_present  # noqa: E402
from nexora.quant import fred, stocks  # noqa: E402
from nexora.research import markets  # noqa: E402
from nexora.research.news import asknews_configured  # noqa: E402

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def main() -> int:
    critical_failures = 0

    # --- Metaculus
    if not os.getenv("METACULUS_TOKEN"):
        line(FAIL, "METACULUS_TOKEN not set")
        critical_failures += 1
    else:
        client = MetaculusClient()
        try:
            user_id = client.get_current_user_id()
            line(OK, f"Metaculus token works (user id {user_id})")
            account = metaculus_account()
            problem = account_problem(account)
            if problem:
                line(FAIL, problem)
                critical_failures += 1
            elif account is not None:
                line(OK, f"Bot account '{account.get('username')}' (API forecasting access: {account.get('api_forecasting_access')})")
            for label, tid in (("Seasonal", client.CURRENT_AI_COMPETITION_ID), ("MiniBench", client.CURRENT_MINIBENCH_ID)):
                try:
                    questions = client.get_all_open_questions_from_tournament(tid)
                    done = sum(1 for q in questions if q.already_forecasted)
                    line(OK, f"{label} ({tid}): {len(questions)} open question(s), {done} already forecast by this bot")
                except Exception as e:  # noqa: BLE001
                    line(WARN, f"{label} ({tid}): could not list questions: {e}")
        except Exception as e:  # noqa: BLE001
            line(FAIL, f"Metaculus token rejected or API unreachable: {e}")
            critical_failures += 1

    # --- OpenRouter
    if not openrouter_key_present():
        line(FAIL, "OPENROUTER_API_KEY not set")
        critical_failures += 1
    else:
        key = os.environ["OPENROUTER_API_KEY"].strip()
        for url in ("https://openrouter.ai/api/v1/key", "https://openrouter.ai/api/v1/auth/key"):
            try:
                r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
                if r.ok:
                    data = r.json().get("data", {})
                    line(OK, f"OpenRouter key valid: usage ${data.get('usage')}, limit {data.get('limit')}, remaining {data.get('limit_remaining')}")
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            line(WARN, "Could not read OpenRouter key info (the key may still work)")

    registry = ModelRegistry.load()
    if registry.available is None:
        line(WARN, "Could not load the OpenRouter model list; fallbacks will be tried at run time")
    for name, profile in PROFILES.items():
        resolved = [f"{s.name}={registry.resolve(s)[0]}" for s in profile.forecasters]
        line(OK, f"profile '{name}': " + ", ".join(resolved))

    if openrouter_key_present():
        helper = PROFILES["lean"].helper
        plan = SlotPlan(helper, registry.resolve(helper))
        try:
            text, model = asyncio.run(call_slot(plan, "Reply with the single word OK."))
            line(OK, f"LLM call works ({model}): {text.strip()[:30]!r}")
        except Exception as e:  # noqa: BLE001
            line(FAIL, f"LLM call failed: {e}")
            critical_failures += 1

    # --- research sources (optional)
    line(OK if asknews_configured() else WARN, "AskNews credentials " + ("found" if asknews_configured() else "missing (news research disabled)"))
    try:
        obs = fred.fetch_series("DGS10")
        line(OK, f"FRED reachable (DGS10 last {obs[-1][0]} = {obs[-1][1]})")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"FRED not reachable: {e}")
    try:
        hist = stocks.fetch_history("SPY")
        line(OK, f"Stock prices reachable via {hist.source} (SPY last close {hist.closes[-1][0]} = {hist.closes[-1][1]:.2f})")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"No stock price source reachable (stock questions fall back to LLM-only): {e}")
    found = markets.search_all(["Federal Reserve rate cut"])
    line(OK if found else WARN, f"Prediction market search returned {len(found)} market(s)")

    print("\nRESULT:", "ready" if critical_failures == 0 else f"{critical_failures} critical problem(s)")
    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
