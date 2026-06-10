"""Provider-agnostic LLM interface.

Skills call ``get_llm(effort=...)`` and use :meth:`LLM.complete`; the provider
SDKs (``anthropic``, ``openai``) are optional extras imported lazily.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from patentkit.llm.routing import ModelChoice, ReasoningEffort, choose_model


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

    def parse_json(self) -> Any:
        """Parse a JSON object/array from the response, tolerating code fences."""
        text = self.text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=0)
        end = max(text.rfind("}"), text.rfind("]")) + 1 or len(text)
        return json.loads(text[start:end])


class LLM:
    """Base class for providers. Subclasses implement :meth:`_complete`."""

    def __init__(self, choice: ModelChoice, api_key: str | None = None):
        self.choice = choice
        self.api_key = api_key

    @property
    def model(self) -> str:
        return self.choice.model

    def complete(
        self,
        prompt: str | list[ChatMessage],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        messages = [ChatMessage("user", prompt)] if isinstance(prompt, str) else prompt
        return self._complete(messages, system=system, max_tokens=max_tokens,
                              temperature=temperature, stop=stop)

    def complete_json(self, prompt: str | list[ChatMessage], **kwargs) -> Any:
        return self.complete(prompt, **kwargs).parse_json()

    def _complete(self, messages: list[ChatMessage], *, system: str | None,
                  max_tokens: int, temperature: float, stop: Optional[list[str]]) -> LLMResponse:
        raise NotImplementedError

    def run_tools(self, messages: list[dict], *, system: str | None = None,
                  tools: Any = (), max_tokens: int = 4096):
        """Run ONE tool-use round: send the neutral-schema conversation with
        the given :class:`~patentkit.llm.tools.ToolDef` list attached and
        return a :class:`~patentkit.llm.tools.ToolRound` (assistant text +
        requested tool calls). The agent loop lives in
        :func:`patentkit.llm.tools.run_tool_loop`, which calls this once per
        round and enforces step/wall-clock budgets."""
        raise NotImplementedError(
            f"The {type(self).__name__} provider does not support tool use. "
            "Use the Anthropic or OpenAI provider (or a test fake implementing "
            "run_tools) to run agentic search."
        )


@dataclass
class UsageTracker:
    """Accumulates token usage across calls; attach via ``get_llm(tracker=...)``."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.by_model[response.model] = self.by_model.get(response.model, 0) + 1


class _TrackedLLM(LLM):
    def __init__(self, inner: LLM, tracker: UsageTracker):
        super().__init__(inner.choice, inner.api_key)
        self._inner = inner
        self._tracker = tracker

    def _complete(self, messages, *, system, max_tokens, temperature, stop) -> LLMResponse:
        response = self._inner._complete(messages, system=system, max_tokens=max_tokens,
                                         temperature=temperature, stop=stop)
        self._tracker.record(response)
        return response

    def run_tools(self, messages, *, system=None, tools=(), max_tokens=4096):
        round_ = self._inner.run_tools(messages, system=system, tools=tools,
                                       max_tokens=max_tokens)
        self._tracker.record(LLMResponse(
            text=round_.text, model=self.model,
            input_tokens=round_.input_tokens, output_tokens=round_.output_tokens,
        ))
        return round_


def get_llm(
    effort: ReasoningEffort | str = ReasoningEffort.MEDIUM,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    tracker: UsageTracker | None = None,
) -> LLM:
    """Build an LLM client for a task's reasoning effort.

    ``model`` overrides the routed default; ``provider`` selects between the
    "anthropic" (default) and "openai" model tables.
    """
    choice = choose_model(effort, provider)
    if model:
        choice = ModelChoice(choice.provider, model, choice.reasoning_effort)

    if choice.provider == "anthropic":
        from patentkit.llm.anthropic_provider import AnthropicLLM
        llm: LLM = AnthropicLLM(choice, api_key=api_key)
    elif choice.provider == "openai":
        from patentkit.llm.openai_provider import OpenAILLM
        llm = OpenAILLM(choice, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {choice.provider!r}")

    return _TrackedLLM(llm, tracker) if tracker else llm
