"""Shared test doubles. Import as ``from tests.fakes import FakeLLM``."""

from __future__ import annotations

import json

from patentkit.llm.base import LLM, ChatMessage, LLMResponse
from patentkit.llm.routing import ModelChoice
from patentkit.llm.tools import ToolCall, ToolRound


class FakeLLM(LLM):
    """Returns canned responses in order, or a constant; records prompts.

    ``responses`` items may be strings or JSON-serializable objects (used by
    :meth:`complete` / :meth:`complete_json`).

    ``tool_script`` scripts the tool-use path (:meth:`run_tools`): a list of
    round dicts like ``{"text": "...", "tool_calls": [{"name": ...,
    "arguments": {...}}]}``. Rounds are returned in order, one per
    ``run_tools`` call; when the script is exhausted, an empty round (no tool
    calls — i.e. "end_turn") is returned. Every ``run_tools`` call records
    the neutral-schema conversation it received in ``tool_conversations`` so
    tests can assert on resumed/injected messages.
    """

    def __init__(self, responses: list | None = None, default: str = "{}",
                 tool_script: list | None = None):
        super().__init__(ModelChoice("fake", "fake-model"))
        self._responses = list(responses or [])
        self._default = default
        self._tool_script = list(tool_script or [])
        self.prompts: list[str] = []
        self.tool_conversations: list[list[dict]] = []
        self.tool_systems: list[str | None] = []
        self._call_counter = 0

    def _complete(self, messages: list[ChatMessage], *, system, max_tokens,
                  temperature, stop) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        if self._responses:
            item = self._responses.pop(0)
        else:
            item = self._default
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(text=text, model="fake-model", input_tokens=1, output_tokens=1)

    def run_tools(self, messages: list[dict], *, system=None, tools=(),
                  max_tokens: int = 4096) -> ToolRound:
        self.tool_conversations.append([dict(m) for m in messages])
        self.tool_systems.append(system)
        if not self._tool_script:
            return ToolRound(text="(no further scripted rounds)", tool_calls=[],
                             input_tokens=1, output_tokens=1)
        item = self._tool_script.pop(0)
        calls = []
        for spec in item.get("tool_calls", []) or []:
            self._call_counter += 1
            calls.append(ToolCall(
                id=spec.get("id", f"call_{self._call_counter}"),
                name=spec["name"],
                arguments=dict(spec.get("arguments") or {}),
            ))
        return ToolRound(text=item.get("text", ""), tool_calls=calls,
                         input_tokens=1, output_tokens=1)
