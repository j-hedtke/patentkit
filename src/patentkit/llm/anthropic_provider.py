"""Anthropic provider (requires the ``patentkit[anthropic]`` extra)."""

from __future__ import annotations

from typing import Optional

from patentkit.config import resolve_key
from patentkit.llm.base import LLM, ChatMessage, LLMResponse


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
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
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
