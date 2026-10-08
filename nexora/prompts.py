"""
Prompt templates.

Design notes (from published FutureEval analyses): explicit base rates and a status-quo
check help; resolution-criteria mistakes and "already resolved" misreadings are the most
expensive errors; prompts that lecture about "Bayesian updating" did worse; the final
answer is kept in a strict machine-readable format.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

NUMERIC_PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

SYSTEM_PROMPT = (
    "You are an elite, well-calibrated forecaster competing in a tournament scored with a "
    "logarithmic scoring rule, where confident mistakes are punished severely. You reason "
    "carefully, rely on base rates and hard evidence, and state uncertainty honestly. "
    "Today's date is {today} (UTC)."
)


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fmt_time(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _days_until(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    days = (value.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 86400
    return f"{days:.1f} days from now"


def question_block(question: MetaculusQuestion) -> str:
    parts = [
        f"QUESTION: {question.question_text}",
        f"Today: {today_str()} (UTC). Forecasting closes: {_fmt_time(question.close_time)}. "
        f"Scheduled resolution: {_fmt_time(question.scheduled_resolution_time)} ({_days_until(question.scheduled_resolution_time)}).",
    ]
    if question.background_info:
        parts.append(f"BACKGROUND:\n{question.background_info.strip()}")
    if question.resolution_criteria:
        parts.append(f"RESOLUTION CRITERIA:\n{question.resolution_criteria.strip()}")
    if question.fine_print:
        parts.append(f"FINE PRINT:\n{question.fine_print.strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- playbooks

_GOOGLE_TRENDS_RE = re.compile(r"google\s+trends?", re.IGNORECASE)
_STOCK_PRICE_RE = re.compile(r"(closing|close)\s+(stock\s+)?price", re.IGNORECASE)
_CP_RE = re.compile(r"community\s+prediction\s+be\s+(higher|lower)\s+than", re.IGNORECASE)
_FRED_RE = re.compile(r"\bFRED\b|fred\.stlouisfed\.org")  # case-sensitive: avoid matching the name "Fred"


def playbooks_for(question: MetaculusQuestion) -> list[str]:
    api = question.api_json if isinstance(question.api_json, dict) else {}
    text = " ".join(
        filter(None, [question.question_text, api.get("title"), question.resolution_criteria, question.fine_print])
    )
    notes: list[str] = []
    if _GOOGLE_TRENDS_RE.search(text):
        notes.append(
            "GOOGLE TRENDS QUESTIONS: A public review of the Fall 2025 MiniBench found these resolved "
            "'decrease' ~46%, 'increase' ~37%, 'no change' ~17% of the time. Read the exact change threshold in "
            "the resolution criteria (it was +/-3 points). Interest usually decays back toward baseline after a spike. "
            "Floor effect: if current interest is already low, it cannot fall by more than the threshold, so "
            "'no change' becomes much more likely. Trends values are relative (0-100) within the chosen window."
        )
    if _STOCK_PRICE_RE.search(text) and isinstance(question, NumericQuestion):
        notes.append(
            "STOCK PRICE QUESTIONS: Over days to weeks, prices are close to a random walk. Anchor on the latest "
            "close, use typical daily volatility (large caps ~1-2%, volatile names 3%+) scaled by the square root of "
            "the number of trading days left, keep the median near the current price, and be careful with units "
            "(e.g. 'in millions of KRW'). Earnings dates widen the distribution but do not predict direction."
        )
    if _STOCK_PRICE_RE.search(text) and isinstance(question, BinaryQuestion) and "higher than" in text.lower():
        notes.append(
            "STOCK DIRECTION QUESTIONS: Day-to-day direction is close to a coin flip; public reviews of MiniBench "
            "found essentially zero skill on these. The only strong signal is how far the price already moved relative "
            "to the reference close (if that close is already known). Stay within a few points of the statistical model."
        )
    if _FRED_RE.search(text) and isinstance(question, NumericQuestion):
        notes.append(
            "FRED DATA QUESTIONS: The answer is one published data point. Anchor on the latest published value, "
            "check the publication lag and whether the date falls on a weekend/holiday, and size the spread from "
            "typical changes over the remaining horizon; scheduled releases or central-bank meetings widen it."
        )
    if _CP_RE.search(text):
        notes.append(
            "COMMUNITY-PREDICTION QUESTIONS: The threshold is usually the CP when this question was created, and "
            "the CP must be strictly higher (or lower) on the date. CPs are sticky and move in small steps; for "
            "'will X happen by <deadline>' questions they tend to drift down as time passes without news. Only "
            "major news moves them sharply."
        )
    return notes


def _extra_context(quant_summary: str | None, playbooks: list[str]) -> str:
    blocks = []
    if quant_summary:
        blocks.append(
            "STATISTICAL MODEL (computed from the underlying data series; treat it as the default forecast and only "
            "deviate for concrete reasons such as a scheduled event the model cannot know about):\n" + quant_summary
        )
    blocks.extend(playbooks)
    return ("\n\n".join(blocks) + "\n\n") if blocks else ""


_SHARED_RULES = """
Important:
- The question has not resolved yet. If the research suggests the outcome already happened, check the dates and
  whether it really satisfies the resolution criteria as written.
- The world usually changes more slowly than news coverage implies; announced plans and deadlines often slip.
- Liquid real-money prediction markets on the same event and deadline are strong evidence; play-money or loosely
  related markets are weak evidence. Check that a market's question and deadline really match.
- Keep the written analysis under ~450 words.
""".strip()


def binary_prompt(question: BinaryQuestion, research: str, quant_summary: str | None) -> str:
    return f"""{question_block(question)}

{_extra_context(quant_summary, playbooks_for(question))}RESEARCH (collected minutes ago; may contain noise or errors):
<research>
{research or "No research available."}
</research>

Estimate the probability that this question resolves YES. Work through, briefly:
1. Resolution mechanics: exactly what must happen, by when, per which source. Note any traps in the fine print.
2. Time left and status quo: what happens if nothing changes? Is anything already in motion?
3. Outside view: 1-3 reference classes with numeric base rates.
4. Inside view: the few pieces of evidence that matter most (including matching prediction markets).
5. Synthesis: start from the base rate, adjust for the specifics, and sanity-check that you are not overconfident.

{_SHARED_RULES}

The last line of your answer must be exactly: "Probability: NN%" with NN between 1 and 99."""


def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str, quant_summary: str | None) -> str:
    option_lines = "\n".join(f"{option}: NN%" for option in question.options)
    return f"""{question_block(question)}

OPTIONS: {question.options}

{_extra_context(quant_summary, playbooks_for(question))}RESEARCH (collected minutes ago; may contain noise or errors):
<research>
{research or "No research available."}
</research>

Estimate the probability of each option. Work through, briefly:
1. Resolution mechanics and which option the status quo implies.
2. Outside view: base rates for each option or for similar situations.
3. Inside view: the evidence that matters most.
4. Synthesis: keep meaningful probability on plausible surprises; never put 0% on an option that could happen.

{_SHARED_RULES}

Finish with a block in exactly this format, using the option names verbatim and summing to 100%:
FINAL PROBABILITIES
{option_lines}"""


def _bound_messages(question: NumericQuestion | DateQuestion) -> str:
    if isinstance(question, DateQuestion):
        upper = question.upper_bound.date().isoformat()
        lower = question.lower_bound.date().isoformat()
        unit = ""
    else:
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        unit = question.unit_of_measure or ""
    upper_msg = (
        f"The question creator thinks the outcome is likely not higher than {upper} {unit} (values above are possible)."
        if question.open_upper_bound
        else f"The outcome cannot be higher than {upper} {unit}."
    )
    lower_msg = (
        f"The question creator thinks the outcome is likely not lower than {lower} {unit} (values below are possible)."
        if question.open_lower_bound
        else f"The outcome cannot be lower than {lower} {unit}."
    )
    return f"{lower_msg}\n{upper_msg}"


def numeric_prompt(question: NumericQuestion, research: str, quant_summary: str | None) -> str:
    unit = question.unit_of_measure or "not stated - infer it from the question"
    lines = "\n".join(f"Percentile {p}: X" for p in NUMERIC_PERCENTILES)
    return f"""{question_block(question)}

UNITS: {unit}
{_bound_messages(question)}

{_extra_context(quant_summary, playbooks_for(question))}RESEARCH (collected minutes ago; may contain noise or errors):
<research>
{research or "No research available."}
</research>

Forecast the distribution of the outcome. Work through, briefly:
1. Resolution mechanics and units (what number, measured how, on which date).
2. The latest known value and its date.
3. Trend and the typical size of changes over a horizon like this one.
4. Expert and market expectations.
5. Low and high tail scenarios.

Calibration: about 90% of outcomes should land between your 5th and 95th percentiles. Forecasters are usually too
narrow; make the tails wide enough to cover surprises.

{_SHARED_RULES}

End with exactly these lines, in the question's units, as plain numbers (no units, no thousands separators, no
scientific notation), strictly increasing:
{lines}"""


def date_prompt(question: DateQuestion, research: str, quant_summary: str | None) -> str:
    lines = "\n".join(f"Percentile {p}: YYYY-MM-DD" for p in NUMERIC_PERCENTILES)
    return f"""{question_block(question)}

{_bound_messages(question)}

{_extra_context(quant_summary, playbooks_for(question))}RESEARCH (collected minutes ago; may contain noise or errors):
<research>
{research or "No research available."}
</research>

Forecast when this will happen. Work through, briefly: resolution mechanics; status quo; base rates for how long
similar things take; the evidence that matters most; early and late scenarios. Timelines usually slip.

{_SHARED_RULES}

End with exactly these lines, dates in chronological order (earliest first):
{lines}"""


def research_prompt(question: MetaculusQuestion) -> str:
    return f"""You are the research assistant of a professional forecaster. Today is {today_str()} (UTC).
Find the most decision-relevant, up-to-date facts for the forecasting question below.

{question_block(question)}

Report in at most ~600 words:
- Current status and the latest developments, each with its date.
- Scheduled events before the resolution date that could decide the outcome.
- Relevant base rates or historical frequencies.
- Latest values of any quantity mentioned (with source and date).
- What experts, officials or markets expect.
Cite sources. Do not give a probability."""


def query_prompt(question: MetaculusQuestion) -> str:
    return f"""Write search queries for this forecasting question. Reply with JSON only, no prose:
{{"news_query": "<natural-language news search, max 12 words>", "market_queries": ["<2-5 keywords>", "<different 2-5 keywords>"]}}

Question: {question.question_text}"""
