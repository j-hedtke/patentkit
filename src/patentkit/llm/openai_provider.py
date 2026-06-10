"""OpenAI provider (requires the ``patentkit[openai]`` extra)."""

from __future__ import annotations

from typing import Optional

from patentkit.config import resolve_key
from patentkit.llm.base import LLM, ChatMessage, LLMResponse


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
