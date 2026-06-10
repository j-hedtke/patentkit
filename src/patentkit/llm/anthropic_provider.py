"""Anthropic provider (requires the ``patentkit[anthropic]`` extra)."""

from __future__ import annotations

from typing import Optional, Sequence

from patentkit.config import resolve_key
from patentkit.llm.base import LLM, ChatMessage, LLMResponse
from patentkit.llm.tools import ToolCall, ToolDef, ToolRound


#: model families where the API removed sampling params (temperature/top_p/top_k
#: return 400): Fable 5 and Opus 4.7+. Sonnet/Haiku 4.x and Opus <=4.6 still accept them.
_NO_SAMPLING_PREFIXES = ("claude-fable", "claude-opus-4-7", "claude-opus-4-8")


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Translate the neutral message schema into Anthropic content blocks."""
    out: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        blocks: list[dict] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif kind == "tool_call":
                blocks.append({"type": "tool_use", "id": block["id"],
                               "name": block["name"], "input": block.get("arguments") or {}})
            elif kind == "tool_result":
                blocks.append({"type": "tool_result",
                               "tool_use_id": block["tool_call_id"],
                               "content": block.get("content", "")})
        out.append({"role": message["role"], "content": blocks})
    return out


class AnthropicLLM(LLM):
    def _client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install the anthropic extra: pip install 'patentkit[anthropic]'") from exc
        return anthropic.Anthropic(api_key=resolve_key("ANTHROPIC_API_KEY", self.api_key))

    def _complete(self, messages: list[ChatMessage], *, system: str | None,
                  max_tokens: int, temperature: float, stop: Optional[list[str]]) -> LLMResponse:
        client = self._client()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        if not self.model.startswith(_NO_SAMPLING_PREFIXES):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        response = client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def run_tools(self, messages: list[dict], *, system: str | None = None,
                  tools: Sequence[ToolDef] = (), max_tokens: int = 4096) -> ToolRound:
        """One Messages-API tool-use round over the neutral conversation."""
        client = self._client()
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=_to_anthropic_messages(messages),
            tools=[{"name": t.name, "description": t.description,
                    "input_schema": t.input_schema} for t in tools],
        )
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in response.content if b.type == "tool_use"
        ]
        return ToolRound(
            text=text,
            tool_calls=calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
