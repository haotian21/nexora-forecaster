"""
NexoraBot: a forecasting-tools `ForecastBot` with

* multi-source research (statistical model / AskNews / Perplexity / prediction markets),
* an ensemble across model families (each "prediction" slot is a different model),
* custom aggregation (log-odds trimmed mean + caps, floored linear pools, CDF pooling),
* comment-first publishing so a forecast is never left without the required comment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    ForecastReport,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    structure_output,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from nexora import prompts
from nexora.aggregation import (
    aggregate_binary,
    aggregate_multiple_choice,
    apply_floor,
    blend_weights,
    pool_cdfs,
    postprocess_binary,
)
from nexora.config import Settings
from nexora.distributions import fit_points_to_range
from nexora.models import ModelRegistry, SlotPlan, call_slot, make_llm
from nexora.parsing import (
    ParseError,
    enforce_increasing,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_numeric_percentiles,
)
from nexora.research import ResearchBundle, gather_research

logger = logging.getLogger(__name__)


def question_key(question: MetaculusQuestion) -> int:
    return question.id_of_question if question.id_of_question is not None else id(question)


class DeferredQuestion(RuntimeError):
    """Raised for questions not started before the run's soft deadline; the next run picks them up."""


class NexoraBot(ForecastBot):
    def __init__(
        self, settings: Settings, registry: ModelRegistry, deadline: float | None = None, **kwargs: Any
    ) -> None:
        self.settings = settings
        self.deadline = deadline  # unix time after which no NEW question is started
        profile = settings.profile
        self.plans = [SlotPlan(slot, registry.resolve(slot)) for slot in profile.forecasters]
        self.helper_plan = SlotPlan(profile.helper, registry.resolve(profile.helper))
        self.web_plan = SlotPlan(profile.web_research, registry.resolve(profile.web_research)) if profile.web_research else None
        self._publish = settings.publish
        self._bundles: dict[int, ResearchBundle] = {}
        self._slot_counters: dict[int, int] = {}
        self._question_semaphore = asyncio.Semaphore(settings.max_concurrent_questions)
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=len(self.plans),
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,  # we publish ourselves, comment first
            folder_to_save_reports_to=kwargs.pop("folder_to_save_reports_to", None),
            skip_previously_forecasted_questions=kwargs.pop("skip_previously_forecasted_questions", True),
            llms={"default": None, "summarizer": None, "researcher": None, "parser": None},
            enable_summarize_research=False,
            extra_metadata_in_explanation=True,
            required_successful_predictions=0.5,
            **kwargs,
        )
        logger.info(
            "Ensemble: " + ", ".join(f"{p.slot.name}={p.primary}({p.slot.effort})" for p in self.plans)
            + f" | helper={self.helper_plan.primary} | web={self.web_plan.primary if self.web_plan else 'off'}"
        )

    # ------------------------------------------------------------------ plumbing

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        return {"default": None, "summarizer": None, "researcher": None, "parser": None}

    def make_llm_dict(self) -> dict[str, Any]:
        return {
            "profile": self.settings.profile.name,
            "forecasters": [f"{p.primary} ({p.slot.effort or 'default'} reasoning)" for p in self.plans],
            "helper": self.helper_plan.primary,
            "web_research": self.web_plan.primary if self.web_plan else None,
        }

    def _next_plan(self, question: MetaculusQuestion) -> SlotPlan:
        # Called synchronously before any await, so each concurrent prediction gets its own slot.
        key = question_key(question)
        index = self._slot_counters.get(key, 0)
        self._slot_counters[key] = index + 1
        return self.plans[index % len(self.plans)]

    def _bundle(self, question: MetaculusQuestion) -> ResearchBundle | None:
        return self._bundles.get(question_key(question))

    def _research_for_prompt(self, question: MetaculusQuestion, research: str) -> tuple[str, str | None]:
        bundle = self._bundle(question)
        if bundle is None:
            return research, None
        return bundle.text, (bundle.quant.summary if bundle.quant else None)

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        self._slot_counters.pop(question_key(question), None)
        async with self._question_semaphore:
            if self.deadline is not None and time.time() > self.deadline:
                raise DeferredQuestion(f"Run time budget used up; {question.page_url} will be forecast next run")
            report = await super()._run_individual_question(question)
        if self._publish:
            await self._publish_report(report)
        self._bundles.pop(question_key(question), None)
        return report

    async def _publish_report(self, report: ForecastReport) -> None:
        """Private comment first, then the forecast. If the forecast post fails the
        question stays 'unforecasted' and is retried on the next run (a duplicate comment
        is harmless; a forecast without a comment would break the prize rules)."""
        question = report.question
        client = self.metaculus_client
        if question.id_of_post is None or question.id_of_question is None:
            raise ValueError(f"Cannot publish: missing ids for {question.page_url}")
        # included_forecast=False: our forecast does not exist yet, so there is nothing to attach.
        await asyncio.to_thread(
            client.post_question_comment, question.id_of_post, report.explanation, is_private=True, included_forecast=False
        )
        prediction = report.prediction
        if isinstance(question, BinaryQuestion):
            await asyncio.to_thread(client.post_binary_question_prediction, question.id_of_question, float(prediction))
        elif isinstance(question, MultipleChoiceQuestion):
            options = {o.option_name: o.probability for o in prediction.predicted_options}
            await asyncio.to_thread(client.post_multiple_choice_question_prediction, question.id_of_question, options)
        elif isinstance(question, (NumericQuestion, DateQuestion)):
            distribution = prediction
            if distribution.cdf_size is None:
                distribution = NumericDistribution.from_question(distribution.declared_percentiles, question)
            cdf = [p.percentile for p in distribution.get_cdf()]
            await asyncio.to_thread(client.post_numeric_question_prediction, question.id_of_question, cdf)
        else:
            await report.publish_report_to_metaculus(metaculus_client=client)
        logger.info(f"Published forecast + comment for {question.page_url}")

    # ------------------------------------------------------------------ research

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        bundle = self._bundle(question)
        sources = ", ".join(bundle.sources) if bundle and bundle.sources else "none"
        return f"Research sources used: {sources}. Full research below."

    async def run_research(self, question: MetaculusQuestion) -> str:
        started = time.time()
        bundle = await gather_research(
            question, self.settings.profile, self.helper_plan, self.web_plan, self.metaculus_client
        )
        if bundle.quant and bundle.quant.kind == "numeric":
            try:
                if not isinstance(question, NumericQuestion) or not bundle.quant.points:
                    raise ValueError("numeric model for a non-numeric question")
                self._make_distribution(bundle.quant.points, question)
            except Exception as e:  # noqa: BLE001 - e.g. units differ from the question: drop the model entirely
                logger.warning(f"Dropping statistical model for {question.page_url}: {e}")
                bundle.quant = None
        self._bundles[question_key(question)] = bundle
        logger.info(
            f"Research for {question.page_url} used {bundle.sources or ['none']} in {time.time() - started:.0f}s"
        )
        header = ""
        if bundle.quant:
            header = f"## Statistical model ({bundle.quant.source})\n{bundle.quant.summary}\n\n"
        return header + bundle.text

    # ------------------------------------------------------------------ forecasts

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        plan = self._next_plan(question)
        text_research, quant_summary = self._research_for_prompt(question, research)
        prompt = prompts.binary_prompt(question, text_research, quant_summary)
        text, model = await call_slot(plan, prompt, prompts.SYSTEM_PROMPT.format(today=prompts.today_str()))
        try:
            probability = parse_binary(text)
        except ParseError:
            parsed: BinaryPrediction = await structure_output(
                text, BinaryPrediction, model=make_llm(self.helper_plan.primary, self.helper_plan.slot, 2)
            )
            probability = float(parsed.prediction_in_decimal)
        probability = min(max(probability, 0.001), 0.999)
        return ReasonedPrediction(prediction_value=probability, reasoning=self._label(model, plan, text))

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        plan = self._next_plan(question)
        text_research, quant_summary = self._research_for_prompt(question, research)
        prompt = prompts.multiple_choice_prompt(question, text_research, quant_summary)
        text, model = await call_slot(plan, prompt, prompts.SYSTEM_PROMPT.format(today=prompts.today_str()))
        try:
            probabilities = parse_multiple_choice(text, list(question.options))
        except ParseError:
            parsed: PredictedOptionList = await structure_output(
                text,
                PredictedOptionList,
                model=make_llm(self.helper_plan.primary, self.helper_plan.slot, 2),
                additional_instructions=(
                    f"Option names must be exactly one of: {question.options}. Include every option; "
                    "use 0 for options given 0%."
                ),
            )
            probabilities = {o.option_name: o.probability for o in parsed.predicted_options}
            if set(probabilities) != set(question.options):
                raise ParseError(f"Parsed options {sorted(probabilities)} do not match {question.options}")
        # forecasting-tools clamps every option to >= 1% and rejects renormalisations that move an
        # option by > 5 points, so floor first (many 0% options would otherwise fail this slot).
        probabilities = apply_floor({o: probabilities[o] for o in question.options}, 0.0101)
        option_list = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=o, probability=probabilities[o]) for o in question.options]
        )
        return ReasonedPrediction(prediction_value=option_list, reasoning=self._label(model, plan, text))

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        plan = self._next_plan(question)
        text_research, quant_summary = self._research_for_prompt(question, research)
        prompt = prompts.numeric_prompt(question, text_research, quant_summary)
        text, model = await call_slot(plan, prompt, prompts.SYSTEM_PROMPT.format(today=prompts.today_str()))
        try:
            points = parse_numeric_percentiles(text, prompts.NUMERIC_PERCENTILES)
        except ParseError:
            parsed: list[Percentile] = await structure_output(
                text,
                list[Percentile],
                model=make_llm(self.helper_plan.primary, self.helper_plan.slot, 2),
                additional_instructions=(
                    f"Values must be in the question's units ({question.unit_of_measure}). Percentiles as fractions "
                    "(0.05 for the 5th). Convert any scientific notation or words like 'million' to plain numbers."
                ),
            )
            points = enforce_increasing(sorted((p.percentile, p.value) for p in parsed))
        distribution = self._make_distribution(points, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=self._label(model, plan, text))

    async def _run_forecast_on_date(self, question: DateQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        plan = self._next_plan(question)
        text_research, quant_summary = self._research_for_prompt(question, research)
        prompt = prompts.date_prompt(question, text_research, quant_summary)
        text, model = await call_slot(plan, prompt, prompts.SYSTEM_PROMPT.format(today=prompts.today_str()))
        try:
            points = parse_date_percentiles(text, prompts.NUMERIC_PERCENTILES)
        except ParseError:
            parsed: list[DatePercentile] = await structure_output(
                text,
                list[DatePercentile],
                model=make_llm(self.helper_plan.primary, self.helper_plan.slot, 2),
                additional_instructions="Percentiles as fractions (0.05 for the 5th). Dates as ISO datetimes, midnight UTC if no time given.",
            )
            points = enforce_increasing(sorted((p.percentile, p.value.timestamp()) for p in parsed))
        distribution = self._make_distribution(points, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=self._label(model, plan, text))

    @staticmethod
    def _label(model: str, plan: SlotPlan, text: str) -> str:
        return f"*Forecaster model: {model} ({plan.slot.effort or 'default'} reasoning)*\n\n{text}"

    @staticmethod
    def _bounds(question: NumericQuestion | DateQuestion) -> tuple[float, float]:
        if isinstance(question, DateQuestion):
            return question.lower_bound.timestamp(), question.upper_bound.timestamp()
        return float(question.lower_bound), float(question.upper_bound)

    def _make_distribution(
        self, points: list[tuple[float, float]], question: NumericQuestion | DateQuestion
    ) -> NumericDistribution:
        lower, upper = self._bounds(question)
        fitted = fit_points_to_range(
            points,
            lower,
            upper,
            open_lower=question.open_lower_bound,
            open_upper=question.open_upper_bound,
            zero_point=getattr(question, "zero_point", None),
        )
        distribution = NumericDistribution.from_question(
            [Percentile(percentile=p, value=v) for p, v in fitted], question
        )
        distribution.get_cdf()  # validate now so a bad distribution fails this slot, not the question
        return distribution

    # ------------------------------------------------------------------ aggregation

    async def _aggregate_predictions(self, predictions: list, question: MetaculusQuestion):
        post = self.settings.post
        bundle = self._bundle(question)
        quant = bundle.quant if bundle else None

        if isinstance(question, BinaryQuestion):
            pooled = aggregate_binary([float(p) for p in predictions], trim=post.trim_binary)
            quant_p = quant.probability if quant and quant.kind == "binary" else None
            final = postprocess_binary(
                pooled,
                quant_probability=quant_p,
                quant_weight=post.quant_weight_binary,
                calib_a=post.calib_a,
                calib_b=post.calib_b,
                floor=post.binary_floor,
                ceiling=post.binary_ceiling,
            )
            logger.info(
                f"{question.page_url}: forecasts {[round(float(p), 3) for p in predictions]} -> pooled {pooled:.3f}"
                f"{f' | quant {quant_p:.3f}' if quant_p is not None else ''} -> final {final:.3f}"
            )
            return final

        if isinstance(question, MultipleChoiceQuestion):
            forecasts = [{o.option_name: o.probability for o in p.predicted_options} for p in predictions]
            pooled = aggregate_multiple_choice(forecasts, list(question.options), floor=max(post.mc_floor, 0.0101))
            return PredictedOptionList(
                predicted_options=[PredictedOption(option_name=o, probability=pooled[o]) for o in question.options]
            )

        if isinstance(question, (NumericQuestion, DateQuestion)):
            cdf_grids = [p.get_cdf() for p in predictions]
            x_axis = [pt.value for pt in cdf_grids[0]]
            cdfs = [[pt.percentile for pt in grid] for grid in cdf_grids]
            weights = blend_weights(len(cdfs), None)
            if quant and quant.kind == "numeric" and quant.points and isinstance(question, NumericQuestion):
                try:
                    quant_dist = self._make_distribution(quant.points, question)
                    quant_grid = quant_dist.get_cdf()
                    if len(quant_grid) == len(x_axis):
                        cdfs.append([pt.percentile for pt in quant_grid])
                        weights = blend_weights(len(cdf_grids), post.quant_weight_numeric)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Could not blend statistical model for {question.page_url}: {e}")
            pooled = pool_cdfs(cdfs, weights)
            return NumericDistribution.from_question(
                [Percentile(value=x, percentile=y) for x, y in zip(x_axis, pooled)], question
            )

        return await super()._aggregate_predictions(predictions, question)
