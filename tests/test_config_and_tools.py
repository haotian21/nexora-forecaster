import importlib.util
import random
from pathlib import Path

import pytest

from nexora.config import PROFILES, ModelSlot, Settings
from nexora.models import ModelRegistry, make_llm


def test_profiles_are_sane():
    for name, profile in PROFILES.items():
        assert profile.forecasters, name
        for slot in profile.forecasters:
            assert slot.candidates and all("/" in c for c in slot.candidates)


def test_every_profile_has_web_research():
    # Fall 2026 MiniBench questions are short-horizon news questions: a profile without a
    # current-news source would forecast them from training data alone.
    assert all(profile.web_research is not None for profile in PROFILES.values())


def test_forecaster_override_and_caps(monkeypatch):
    monkeypatch.setenv("NEXORA_FORECASTERS", "anthropic/claude-opus-5.5:high, openai/gpt-6-sol:medium")
    monkeypatch.setenv("NEXORA_BINARY_CAP", "0.03")
    settings = Settings.from_env(publish=False, profile_name="lean")
    assert [s.candidates[0] for s in settings.profile.forecasters] == ["anthropic/claude-opus-5.5", "openai/gpt-6-sol"]
    assert settings.profile.forecasters[1].effort == "medium"
    assert settings.post.binary_floor == pytest.approx(0.03) and settings.post.binary_ceiling == pytest.approx(0.97)
    monkeypatch.setenv("NEXORA_BINARY_CAP", "0.7")
    with pytest.raises(ValueError):
        Settings.from_env(publish=False)


def test_unknown_profile(monkeypatch):
    with pytest.raises(ValueError):
        Settings.from_env(profile_name="turbo")


def test_registry_resolution():
    slot = ModelSlot("x", ("a/one", "b/two", "c/three"))
    assert ModelRegistry(None).resolve(slot) == ["a/one", "b/two", "c/three"]
    assert ModelRegistry({"b/two", "c/three"}).resolve(slot) == ["b/two", "c/three"]
    # nothing listed -> still try the configured order rather than giving up
    assert ModelRegistry({"z/zzz"}).resolve(slot) == ["a/one", "b/two", "c/three"]


def test_make_llm_sends_reasoning_via_extra_body():
    llm = make_llm("anthropic/claude-opus-5.5", ModelSlot("s", ("x/y",), effort="high", max_tokens=1000), 1)
    assert llm.litellm_kwargs["extra_body"] == {"reasoning": {"effort": "high"}}
    assert llm.litellm_kwargs["model"] == "openrouter/anthropic/claude-opus-5.5"
    no_effort = make_llm("perplexity/sonar-pro", ModelSlot("w", ("x/y",), effort=None), 1)
    assert "extra_body" not in no_effort.litellm_kwargs


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_calibration_detects_overconfidence():
    fc = _load_script("fit_calibration")
    rng = random.Random(0)
    pairs = []
    for _ in range(400):
        true_p = rng.uniform(0.05, 0.95)
        y = int(rng.random() < true_p)
        # overconfident forecaster: doubles the log-odds
        from nexora.aggregation import logit, sigmoid

        pairs.append((sigmoid(2.0 * logit(true_p)), y))
    a, b = fc.fit(pairs)
    assert a < 0.8  # recalibration should shrink toward 50%
    assert fc.log_loss(pairs, a, b) < fc.log_loss(pairs)
