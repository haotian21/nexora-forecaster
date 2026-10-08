"""
Fit Platt recalibration (z' = a*z + b on the log-odds scale) from this bot's own RESOLVED
binary forecasts, and report whether it would have improved the log score.

  uv run python scripts/fit_calibration.py                 # seasonal tournament + MiniBench
  uv run python scripts/fit_calibration.py --tournament minibench

Only resolved questions are used, which is allowed by the rules (tuning on open questions is not).
If the suggestion is clearly better and based on enough questions (>= 60), set the repository
variables NEXORA_CALIB_A / NEXORA_CALIB_B. Otherwise leave the defaults (a=1, b=0).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dotenv  # noqa: E402

dotenv.load_dotenv()

import numpy as np  # noqa: E402
from forecasting_tools import BinaryQuestion, MetaculusClient  # noqa: E402
from forecasting_tools.helpers.metaculus_client import ApiFilter  # noqa: E402

from nexora.aggregation import logit, sigmoid  # noqa: E402


def fetch_pairs(client: MetaculusClient, tournament) -> list[tuple[float, int]]:
    api_filter = ApiFilter(
        allowed_tournaments=[tournament],
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        is_previously_forecasted_by_user=True,
        group_question_mode="unpack_subquestions",
    )
    questions = asyncio.run(
        client.get_questions_matching_filter(api_filter, num_questions=2000, error_if_question_target_missed=False)
    )
    pairs = []
    for q in questions:
        if not isinstance(q, BinaryQuestion) or not q.previous_forecasts:
            continue
        resolution = q.binary_resolution
        if not isinstance(resolution, bool):
            continue  # annulled / ambiguous
        last = q.previous_forecasts[-1].prediction_in_decimal
        pairs.append((float(last), int(resolution)))
    return pairs


def log_loss(pairs, a=1.0, b=0.0) -> float:
    total = 0.0
    for p, y in pairs:
        q = min(max(sigmoid(a * logit(p) + b), 1e-4), 1 - 1e-4)
        total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return total / len(pairs)


def fit(pairs, l2: float = 2.0) -> tuple[float, float]:
    """Grid search with a ridge penalty pulling (a, b) toward (1, 0) - robust for small samples."""
    best = (1.0, 0.0)
    best_obj = float("inf")
    n = len(pairs)
    for a in np.arange(0.5, 2.01, 0.05):
        for b in np.arange(-0.6, 0.61, 0.05):
            obj = log_loss(pairs, a, b) + l2 / n * ((a - 1) ** 2 + b**2)
            if obj < best_obj:
                best_obj, best = obj, (float(a), float(b))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", default=None, help="tournament id or slug (default: seasonal + MiniBench)")
    args = parser.parse_args()
    client = MetaculusClient()
    tournaments = [args.tournament] if args.tournament else [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    pairs: list[tuple[float, int]] = []
    for t in tournaments:
        found = fetch_pairs(client, t)
        print(f"{t}: {len(found)} resolved binary forecasts")
        pairs.extend(found)
    if len(pairs) < 20:
        print("Not enough resolved questions yet (need 20+, ideally 60+).")
        return 0
    a, b = fit(pairs)
    base, tuned = log_loss(pairs), log_loss(pairs, a, b)
    yes_rate = sum(y for _, y in pairs) / len(pairs)
    mean_p = sum(p for p, _ in pairs) / len(pairs)
    print(f"Questions: {len(pairs)} | resolved YES rate {yes_rate:.1%} | mean forecast {mean_p:.1%}")
    print(f"Log loss now: {base:.4f} | with a={a:.2f}, b={b:+.2f}: {tuned:.4f} (in-sample)")
    if len(pairs) >= 60 and tuned < base - 0.01:
        print(f"Suggestion: set repository variables NEXORA_CALIB_A={a:.2f} and NEXORA_CALIB_B={b:.2f}")
        print("(If you already changed them before: new A = A_old * a, new B = a * B_old + b, because the "
              "forecasts above were already recalibrated with the old values.)")
    else:
        print("Suggestion: keep the defaults (improvement too small or too few questions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
