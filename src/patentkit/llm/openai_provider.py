"""OpenAI provider (requires the ``patentkit[openai]`` extra)."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from patentkit.config import resolve_key
from patentkit.llm.base import LLM, ChatMessage, LLMResponse
from patentkit.llm.tools import ToolCall, ToolDef, ToolRound


def _to_responses_input(messages: list[dict]) -> list[dict]:
    """Translate the neutral message schema into Responses-API input items.

    The conversation is rebuilt as accumulated input on every round (rather
    than ``previous_response_id`` chaining) so resumed/feedback-injected
    conversations work without server-side state.
    """
    items: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            items.append({"role": message["role"], "content": content})
            continue
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                items.append({"role": message["role"], "content": block["text"]})
            elif kind == "tool_call":
                items.append({"type": "function_call", "call_id": block["id"],
                              "name": block["name"],
                              "arguments": json.dumps(block.get("arguments") or {})})
            elif kind == "tool_result":
                items.append({"type": "function_call_output",
                              "call_id": block["tool_call_id"],
                              "output": block.get("content", "")})
    return items


class OpenAILLM(LLM):
    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install the openai extra: pip install 'patentkit[openai]'") from exc
        return openai.OpenAI(api_key=resolve_key("OPENAI_API_KEY", self.api_key))

    def _complete(self, messages: list[ChatMessage], *, system: str | None,
                  max_tokens: int, temperature: float, stop: Optional[list[str]]) -> LLMResponse:
        client = self._client()
        input_items = []
        if system:
            input_items.append({"role": "developer", "content": system})
        input_items += [{"role": m.role, "content": m.content} for m in messages]
        kwargs = dict(model=self.model, input=input_items, max_output_tokens=max_tokens)
        if self.choice.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.choice.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.output_text,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=response,
        )

    def run_tools(self, messages: list[dict], *, system: str | None = None,
                  tools: Sequence[ToolDef] = (), max_tokens: int = 4096) -> ToolRound:
        """One Responses-API function-tool round over the neutral conversation."""
        client = self._client()
        input_items: list[dict] = []
        if system:
            input_items.append({"role": "developer", "content": system})
        input_items += _to_responses_input(messages)
        kwargs = dict(
            model=self.model,
            input=input_items,
            max_output_tokens=max_tokens,
            tools=[{"type": "function", "name": t.name, "description": t.description,
                    "parameters": t.input_schema} for t in tools],
        )
        if self.choice.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.choice.reasoning_effort}
        response = client.responses.create(**kwargs)
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", "") == "function_call":
                try:
                    arguments = json.loads(item.arguments) if item.arguments else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": item.arguments}
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments))
        usage = getattr(response, "usage", None)
        return ToolRound(
            text=getattr(response, "output_text", "") or "",
            tool_calls=calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
