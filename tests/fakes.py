"""Shared test doubles. Import as ``from tests.fakes import FakeLLM``."""

from __future__ import annotations

import json

from patentkit.llm.base import LLM, ChatMessage, LLMResponse
from patentkit.llm.routing import ModelChoice


class FakeLLM(LLM):
    """Returns canned responses in order, or a constant; records prompts.

    ``responses`` items may be strings or JSON-serializable objects.
    """

    def __init__(self, responses: list | None = None, default: str = "{}"):
        super().__init__(ModelChoice("fake", "fake-model"))
        self._responses = list(responses or [])
        self._default = default
        self.prompts: list[str] = []

    def _complete(self, messages: list[ChatMessage], *, system, max_tokens,
                  temperature, stop) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        if self._responses:
            item = self._responses.pop(0)
        else:
            item = self._default
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(text=text, model="fake-model", input_tokens=1, output_tokens=1)
