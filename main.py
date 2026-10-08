"""
Entry point.

  uv run python main.py --mode tournament        # Fall 2026 seasonal tournament + MiniBench (default)
  uv run python main.py --mode seasonal          # seasonal tournament only
  uv run python main.py --mode minibench         # MiniBench only
  uv run python main.py --mode test              # bot-testing-area (safe place to check the setup)
  uv run python main.py --mode test --dry-run    # same, but nothing is posted

Rules reminder: tournament forecasts must be fully automated. Do not run the bot on open
tournament questions, read the output and then tweak it; iterate on the bot-testing-area or
on resolved questions instead.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import warnings

import dotenv

dotenv.load_dotenv()
warnings.filterwarnings("ignore", message=r".*does not support cost tracking.*")

from forecasting_tools import ForecastReport, MetaculusClient  # noqa: E402

from nexora.account import account_problem, metaculus_account  # noqa: E402
from nexora.bot import DeferredQuestion, NexoraBot  # noqa: E402
from nexora.config import PROFILES, Settings  # noqa: E402
from nexora.models import ModelRegistry, openrouter_key_present  # noqa: E402

logger = logging.getLogger("nexora")

TEST_TOURNAMENT = "bot-testing-area"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "httpx", "httpcore", "openai", "forecasting_tools.ai_models.model_tracker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # forecasting-tools warns on every multiple-choice list whose float sum is not exactly 1.0
    logging.getLogger("forecasting_tools.data_models.multiple_choice_report").setLevel(logging.ERROR)


def check_environment(publish: bool) -> list[str]:
    problems = []
    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token or token == "REPLACE_ME":
        problems.append("METACULUS_TOKEN is missing (create a bot account at https://www.metaculus.com/futureeval/participate/)")
    if not openrouter_key_present():
        problems.append("OPENROUTER_API_KEY is missing (free credits: https://forms.gle/aQdYMq9Pisrf1v7d8)")
    if publish and not problems:
        # A personal account's token would comment on every question and have every forecast refused.
        problem = account_problem(metaculus_account(token))
        if problem:
            problems.append(problem)
    return problems


async def run_target(bot: NexoraBot, tournament_id: int | str, limit: int | None) -> list[ForecastReport | BaseException]:
    questions = bot.metaculus_client.get_all_open_questions_from_tournament(tournament_id)
    logger.info(f"{tournament_id}: {len(questions)} open question(s)")
    if limit is not None:
        pending = [q for q in questions if not (bot.skip_previously_forecasted_questions and q.already_forecasted)]
        questions = pending[:limit]
    return await bot.forecast_questions(questions, return_exceptions=True)


def _is_deferred(error: BaseException) -> bool:
    return isinstance(error, DeferredQuestion) or "Run time budget used up" in str(error)


def summarise(label: str, reports: list[ForecastReport | BaseException]) -> tuple[int, int]:
    ok = [r for r in reports if isinstance(r, ForecastReport)]
    deferred = [r for r in reports if isinstance(r, BaseException) and _is_deferred(r)]
    failed = [r for r in reports if isinstance(r, BaseException) and not _is_deferred(r)]
    cost = sum(r.price_estimate or 0 for r in ok)
    print(
        f"\n=== {label}: {len(ok)} forecast(s), {len(failed)} failure(s), {len(deferred)} deferred to next run, "
        f"est. LLM cost ${cost:.2f} ==="
    )
    for r in ok:
        print(f"  OK   {r.question.page_url}  ->  {type(r).make_readable_prediction(r.prediction)[:120]!r}")
    for e in failed:
        print(f"  FAIL {e.__class__.__name__}: {str(e)[:300]}")
    return len(ok), len(failed)


async def main_async(args: argparse.Namespace) -> int:
    publish = not args.dry_run
    problems = check_environment(publish)
    if problems:
        for p in problems:
            print(f"Setup problem: {p}")
        return 2

    registry = ModelRegistry.load()
    client = MetaculusClient()

    def settings_for(profile_name: str | None) -> Settings:
        return Settings.from_env(publish=publish, profile_name=profile_name)

    # Tournament ids rotate every season; forecasting-tools ships the current ones, env vars can override.
    seasonal_id = os.getenv("NEXORA_SEASONAL_TOURNAMENT") or client.CURRENT_AI_COMPETITION_ID
    minibench_id = os.getenv("NEXORA_MINIBENCH_TOURNAMENT") or client.CURRENT_MINIBENCH_ID
    targets: list[tuple[str, int | str, Settings, bool]] = []
    if args.mode in ("tournament", "seasonal"):
        targets.append(("Seasonal tournament", seasonal_id, settings_for(args.profile), True))
    if args.mode in ("tournament", "minibench"):
        mb_profile = args.minibench_profile or os.getenv("NEXORA_MINIBENCH_PROFILE") or args.profile
        targets.append(("MiniBench", minibench_id, settings_for(mb_profile), True))
    if args.mode == "test":
        targets.append(("Bot testing area", TEST_TOURNAMENT, settings_for(args.profile), False))

    if args.dry_run and args.mode != "test":
        print("Refusing --dry-run on live tournaments: reading outputs on open tournament questions and then "
              "changing the bot breaks the no-human-in-the-loop rule. Use --mode test --dry-run instead.")
        return 2

    deadline = time.time() + 60 * float(os.getenv("NEXORA_MAX_RUN_MINUTES", "38"))
    total_ok = total_failed = 0
    for label, tournament_id, settings, skip_done in targets:
        bot = NexoraBot(
            settings, registry, deadline=deadline, metaculus_client=client, skip_previously_forecasted_questions=skip_done
        )
        try:
            reports = await run_target(bot, tournament_id, args.limit)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"{label}: could not run: {e}")
            total_failed += 1
            continue
        ok, failed = summarise(label, reports)
        total_ok += ok
        total_failed += failed

    if total_failed and not total_ok:
        return 1  # everything failed: make the GitHub run red so it gets noticed
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexora forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "seasonal", "minibench", "test"], default="tournament")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="overrides NEXORA_PROFILE")
    parser.add_argument("--minibench-profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--dry-run", action="store_true", help="do not post anything (test mode only)")
    parser.add_argument("--limit", type=int, default=None, help="max new questions per target (for cheap tests)")
    args = parser.parse_args()
    configure_logging()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
