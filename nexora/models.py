"""
Model access through OpenRouter.

* `ModelRegistry` checks which of our preferred model ids OpenRouter currently serves
  (public endpoint, no key needed), so retired or renamed models are skipped.
* `call_slot` invokes a slot's model with its reasoning effort and falls back to the
  next candidate if a call fails, so one provider outage does not cost a question.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import requests
from forecasting_tools import GeneralLlm

from nexora.config import ModelSlot

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class ModelRegistry:
    def __init__(self, available: set[str] | None) -> None:
        # None means "unknown" (list could not be fetched): trust the configured order.
        self.available = available

    @classmethod
    def load(cls, timeout: int = 20) -> "ModelRegistry":
        try:
            response = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
            response.raise_for_status()
            ids = {item["id"] for item in response.json().get("data", []) if "id" in item}
            if not ids:
                raise ValueError("empty model list")
            logger.info(f"OpenRouter lists {len(ids)} models")
            return cls(ids)
        except Exception as e:  # noqa: BLE001 - network problems must not stop the bot
            logger.warning(f"Could not load OpenRouter model list ({e}); using configured model order as-is")
            return cls(None)

    def resolve(self, slot: ModelSlot) -> list[str]:
        if self.available is None:
            return list(slot.candidates)
        resolved = [m for m in slot.candidates if m in self.available]
        if not resolved:
            logger.warning(
                f"None of the candidates for slot '{slot.name}' are listed on OpenRouter: {slot.candidates}. "
                "Trying them anyway."
            )
            return list(slot.candidates)
        return resolved


@dataclass
class SlotPlan:
    slot: ModelSlot
    models: list[str]  # resolved, in fallback order

    @property
    def primary(self) -> str:
        return self.models[0]


def make_llm(model_id: str, slot: ModelSlot, allowed_tries: int) -> GeneralLlm:
    kwargs: dict = {
        "model": f"openrouter/{model_id}",
        "temperature": None,  # reasoning models manage their own sampling
        "timeout": slot.timeout,
        "allowed_tries": allowed_tries,
        "max_tokens": slot.max_tokens,
    }
    if slot.effort:
        # OpenRouter's unified reasoning parameter; sent via extra_body so LiteLLM never drops it.
        kwargs["extra_body"] = {"reasoning": {"effort": slot.effort}}
    return GeneralLlm(**kwargs)


class LlmCallError(RuntimeError):
    pass


async def call_slot(plan: SlotPlan, prompt: str, system_prompt: str | None = None) -> tuple[str, str]:
    """Returns (text, model_id_used). Tries each resolved model once more on failure."""
    errors: list[str] = []
    for index, model_id in enumerate(plan.models):
        llm = make_llm(model_id, plan.slot, allowed_tries=2 if index == 0 else 1)
        started = time.time()
        try:
            text = await llm.invoke(prompt, system_prompt=system_prompt)
            if not text or not text.strip():
                raise LlmCallError("empty response")
            logger.debug(f"{plan.slot.name} -> {model_id} answered in {time.time() - started:.0f}s")
            return text, model_id
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            message = f"{model_id}: {e.__class__.__name__}: {str(e)[:300]}"
            logger.warning(f"Slot '{plan.slot.name}' model failed -> {message}")
            errors.append(message)
    raise LlmCallError(f"All models failed for slot '{plan.slot.name}': {errors}")


def openrouter_key_present() -> bool:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return bool(key) and key not in {"REPLACE_ME", "your-api-key-here"}
