"""Default model choices, routed by the reasoning effort a task needs.

patentkit tasks declare a :class:`ReasoningEffort` rather than a model id, so
swapping providers (or upgrading model generations) is one table edit. Users
override per-call (``get_llm(model="...")``) or globally by mutating
:data:`DEFAULT_MODELS`.

Effort tiers used across the toolkit:

- LOW    — extraction, formatting, keyword voting, classification
- MEDIUM — claim interpretation, passage selection, summarization/ranking
- HIGH   — disclosure assessment, infringement reasoning, search planning,
           claim charting, drafting
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ModelChoice:
    provider: str  # "anthropic" | "openai"
    model: str
    #: provider-native effort knob, where supported (OpenAI reasoning models)
    reasoning_effort: str | None = None


#: provider -> effort -> default model
DEFAULT_MODELS: dict[str, dict[ReasoningEffort, ModelChoice]] = {
    "anthropic": {
        ReasoningEffort.LOW: ModelChoice("anthropic", "claude-haiku-4-5"),
        ReasoningEffort.MEDIUM: ModelChoice("anthropic", "claude-sonnet-4-6"),
        ReasoningEffort.HIGH: ModelChoice("anthropic", "claude-fable-5"),
    },
    "openai": {
        ReasoningEffort.LOW: ModelChoice("openai", "gpt-5-mini", reasoning_effort="low"),
        ReasoningEffort.MEDIUM: ModelChoice("openai", "gpt-5.1", reasoning_effort="medium"),
        ReasoningEffort.HIGH: ModelChoice("openai", "gpt-5.1", reasoning_effort="high"),
    },
}

#: alternates kept current so users can swap one line; opus is a strong
#: anthropic HIGH alternative when fable access is unavailable.
ALTERNATE_MODELS: dict[str, ModelChoice] = {
    "anthropic-high-opus": ModelChoice("anthropic", "claude-opus-4-8"),
}

DEFAULT_PROVIDER = "anthropic"


def choose_model(
    effort: ReasoningEffort | str = ReasoningEffort.MEDIUM,
    provider: str | None = None,
) -> ModelChoice:
    effort = ReasoningEffort(effort)
    provider = provider or DEFAULT_PROVIDER
    try:
        return DEFAULT_MODELS[provider][effort]
    except KeyError:
        raise ValueError(
            f"No default model for provider={provider!r}, effort={effort.value!r}. "
            f"Known providers: {sorted(DEFAULT_MODELS)}"
        ) from None
