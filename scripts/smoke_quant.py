"""
Live smoke test of the statistical models (no LLM calls, no cost): builds three fake
MiniBench-style questions and runs detection -> data download -> model for each.

  uv run python scripts/smoke_quant.py

FRED and the stock price sources (Yahoo, Nasdaq fallback) need internet; the
community-prediction model also needs METACULUS_TOKEN.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dotenv  # noqa: E402

dotenv.load_dotenv()

from forecasting_tools import BinaryQuestion, MetaculusClient, NumericQuestion  # noqa: E402

from nexora.quant import run_quant  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def next_weekday(days_ahead: int) -> str:
    d = datetime.now(timezone.utc).date() + timedelta(days=days_ahead)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def last_weekday() -> str:
    d = datetime.now(timezone.utc).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


async def main() -> None:
    target = next_weekday(7)
    fred_q = NumericQuestion(
        question_text=f"What will be the value of 'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity' on {target}?",
        api_json={"title": "What will the value of FRED series DGS10 be?"},
        upper_bound=6.0, lower_bound=2.0, open_upper_bound=True, open_lower_bound=True, unit_of_measure="Percent",
    )
    stock_q = BinaryQuestion(
        question_text=f"Will AAPL's market close price on {target} be higher than its market close price on {last_weekday()}?",
    )
    cp_q = BinaryQuestion(
        question_text=f"Will the community prediction be higher than 70.00% on {target} for the Metaculus question "
        "'Will the S&P 500 close above 7,500 at the end of 2026?'?",
        resolution_criteria="Linked question: https://www.metaculus.com/questions/41130/",
    )
    client = MetaculusClient()
    has_token = bool(os.getenv("METACULUS_TOKEN", "").strip())
    for label, question in (("FRED numeric", fred_q), ("Stock direction", stock_q), ("Community prediction", cp_q)):
        if question is cp_q and not has_token:
            print(f"\n=== {label} ===\nskipped: set METACULUS_TOKEN in .env to test this one")
            continue
        result = await run_quant(question, metaculus_client=client)
        print(f"\n=== {label} ===")
        if result is None:
            print("no result (see warning above: network, token or template mismatch)")
        else:
            print(f"source={result.source} kind={result.kind} probability={result.probability}")
            print(result.summary)


if __name__ == "__main__":
    asyncio.run(main())
