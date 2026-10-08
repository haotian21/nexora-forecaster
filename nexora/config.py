"""
Central configuration: model ensembles ("profiles"), post-processing parameters and
tournament targets. Everything can be overridden with environment variables so the
bot can be tuned from GitHub repository variables without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelSlot:
    """One member of the ensemble.

    `candidates` are OpenRouter model ids in order of preference. At startup the bot
    keeps the first one that OpenRouter currently lists, so a renamed or retired model
    degrades gracefully instead of crashing the run.
    """

    name: str
    candidates: tuple[str, ...]
    effort: str | None = "high"  # OpenRouter unified reasoning effort
    max_tokens: int = 24000
    timeout: int = 420


# ---- Candidate lists (OpenRouter ids, checked against the live model list) ----
CLAUDE_TOP = ("anthropic/claude-fable-5.1", "~anthropic/claude-fable-latest", "anthropic/claude-opus-5.5")
CLAUDE_MAIN = (
    "anthropic/claude-opus-5.5",
    "~anthropic/claude-opus-latest",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
)
CLAUDE_MID = ("anthropic/claude-sonnet-5", "~anthropic/claude-sonnet-latest", "anthropic/claude-sonnet-4.6")
OPENAI_TOP = ("openai/gpt-6-astra", "openai/gpt-5.5", "openai/gpt-6-sol")
OPENAI_MAIN = ("openai/gpt-6-sol", "openai/gpt-5.5", "openai/gpt-5.4")
GEMINI_MAIN = ("google/gemini-3.1-pro-preview", "~google/gemini-pro-latest", "google/gemini-3.8-flash")
CHEAP_HELPER = (
    "anthropic/claude-haiku-4.5",
    "~anthropic/claude-haiku-latest",
    "openai/gpt-6-luna",
    "openai/gpt-5.4-mini",
    "google/gemini-3.8-flash",
)
PERPLEXITY_MAIN = ("perplexity/sonar-pro", "perplexity/sonar-reasoning-pro", "perplexity/sonar")


@dataclass(frozen=True)
class Profile:
    name: str
    forecasters: tuple[ModelSlot, ...]
    helper: ModelSlot  # parsing, query generation, extraction (cheap, reliable)
    web_research: ModelSlot | None  # search-grounded research model (Perplexity)
    use_asknews: bool = True
    # AskNews archive search (160 days) costs 5 of the free tier's ~1k monthly calls; latest news costs 1.
    asknews_archive: bool = True
    use_markets: bool = True
    use_quant: bool = True
    use_similar: bool = True
    max_research_chars: int = 28000
    description: str = ""


def _helper() -> ModelSlot:
    return ModelSlot("helper", CHEAP_HELPER, effort="low", max_tokens=6000, timeout=180)


def _web() -> ModelSlot:
    return ModelSlot("web", PERPLEXITY_MAIN, effort=None, max_tokens=6000, timeout=240)


PROFILES: dict[str, Profile] = {
    "lean": Profile(
        name="lean",
        description="3 forecasters, medium reasoning, Perplexity web search. Roughly $0.2-0.4 per question.",
        forecasters=(
            ModelSlot("claude-1", CLAUDE_MAIN, effort="medium", max_tokens=16000),
            ModelSlot("openai-1", OPENAI_MAIN, effort="medium", max_tokens=16000),
            ModelSlot("claude-2", CLAUDE_MID, effort="medium", max_tokens=16000),
        ),
        helper=_helper(),
        # Fall 2026 MiniBench asks about this week's news; without web search (or AskNews)
        # the forecasters would only have their training data.
        web_research=_web(),
        asknews_archive=False,  # short-horizon questions: the latest 48 hours of news matter most
    ),
    "standard": Profile(
        name="standard",
        description="5 forecasters across 3 model families, high reasoning, Perplexity + AskNews + markets. Roughly $0.6-1.2 per question.",
        forecasters=(
            ModelSlot("claude-1", CLAUDE_MAIN, effort="high"),
            ModelSlot("claude-2", CLAUDE_MAIN, effort="high"),
            ModelSlot("openai-1", OPENAI_MAIN, effort="high"),
            ModelSlot("openai-2", OPENAI_MAIN, effort="high"),
            ModelSlot("gemini-1", GEMINI_MAIN, effort="high"),
        ),
        helper=_helper(),
        web_research=_web(),
    ),
    "max": Profile(
        name="max",
        description="5 top-tier forecasters (Fable / GPT-6 Astra / Opus), high reasoning. Roughly $2-4 per question.",
        forecasters=(
            ModelSlot("claude-top-1", CLAUDE_TOP, effort="high", max_tokens=32000),
            ModelSlot("claude-top-2", CLAUDE_TOP, effort="high", max_tokens=32000),
            ModelSlot("openai-top-1", OPENAI_TOP, effort="high", max_tokens=32000),
            ModelSlot("openai-top-2", OPENAI_TOP, effort="high", max_tokens=32000),
            ModelSlot("claude-1", CLAUDE_MAIN, effort="high", max_tokens=32000),
        ),
        helper=_helper(),
        web_research=_web(),
    ),
    "claude-only": Profile(
        name="claude-only",
        description="Anthropic models only (Fable + Opus + Sonnet). Roughly $0.8-2 per question.",
        forecasters=(
            ModelSlot("claude-top-1", CLAUDE_TOP, effort="high", max_tokens=32000),
            ModelSlot("claude-1", CLAUDE_MAIN, effort="high"),
            ModelSlot("claude-2", CLAUDE_MAIN, effort="high"),
            ModelSlot("claude-3", CLAUDE_MID, effort="high"),
        ),
        helper=_helper(),
        web_research=_web(),
    ),
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_forecaster_override(raw: str) -> tuple[ModelSlot, ...]:
    """NEXORA_FORECASTERS="anthropic/claude-opus-5.5:high,openai/gpt-6-sol:medium"."""
    slots = []
    for i, item in enumerate(x.strip() for x in raw.split(",") if x.strip()):
        model, _, effort = item.partition(":")
        # Allow "~anthropic/..." aliases and ":batch"-free ids only.
        slots.append(ModelSlot(f"custom-{i + 1}", (model.strip(),), effort=(effort.strip() or "high")))
    if not slots:
        raise ValueError("NEXORA_FORECASTERS was set but contained no models")
    return tuple(slots)


@dataclass(frozen=True)
class PostProcessing:
    binary_floor: float = 0.02  # never forecast below 2% / above 98% (log score insurance)
    binary_ceiling: float = 0.98
    mc_floor: float = 0.01  # every multiple-choice option keeps at least 1%
    calib_a: float = 1.0  # Platt recalibration z' = a*z + b on the log-odds scale
    calib_b: float = 0.0
    trim_binary: bool = True  # drop the most extreme forecast on each side (n>=5)
    quant_weight_numeric: float = 0.6  # weight of the statistical model for data-series questions
    quant_weight_binary: float = 0.7


@dataclass(frozen=True)
class Settings:
    profile: Profile
    post: PostProcessing = field(default_factory=PostProcessing)
    max_concurrent_questions: int = 4
    publish: bool = True
    request_timeout_seconds: int = 25  # for the small public HTTP APIs

    @classmethod
    def from_env(cls, publish: bool | None = None, profile_name: str | None = None) -> "Settings":
        profile_name = (profile_name or os.getenv("NEXORA_PROFILE") or "standard").strip().lower()
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown NEXORA_PROFILE '{profile_name}'. Options: {sorted(PROFILES)}")
        profile = PROFILES[profile_name]
        override = os.getenv("NEXORA_FORECASTERS", "").strip()
        if override:
            profile = replace(profile, name=f"{profile.name}+custom", forecasters=_parse_forecaster_override(override))
        profile = replace(
            profile,
            use_asknews=_env_bool("NEXORA_USE_ASKNEWS", profile.use_asknews),
            use_markets=_env_bool("NEXORA_USE_MARKETS", profile.use_markets),
            use_quant=_env_bool("NEXORA_USE_QUANT", profile.use_quant),
            use_similar=_env_bool("NEXORA_USE_SIMILAR", profile.use_similar),
            web_research=profile.web_research if _env_bool("NEXORA_USE_WEB", profile.web_research is not None) else None,
        )
        cap = _env_float("NEXORA_BINARY_CAP", 0.02)
        post = PostProcessing(
            binary_floor=cap,
            binary_ceiling=1.0 - cap,
            mc_floor=_env_float("NEXORA_MC_FLOOR", 0.01),
            calib_a=_env_float("NEXORA_CALIB_A", 1.0),
            calib_b=_env_float("NEXORA_CALIB_B", 0.0),
            quant_weight_numeric=_env_float("NEXORA_QUANT_WEIGHT_NUMERIC", 0.6),
            quant_weight_binary=_env_float("NEXORA_QUANT_WEIGHT_BINARY", 0.7),
        )
        if not (0.0 < post.binary_floor < 0.5):
            raise ValueError("NEXORA_BINARY_CAP must be between 0 and 0.5")
        return cls(
            profile=profile,
            post=post,
            max_concurrent_questions=_env_int("NEXORA_MAX_CONCURRENT_QUESTIONS", 4),
            publish=publish if publish is not None else _env_bool("NEXORA_PUBLISH", True),
        )
