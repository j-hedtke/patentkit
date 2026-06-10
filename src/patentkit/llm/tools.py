"""Provider-agnostic tool-use runtime.

The agentic search core (``patentkit.agents.agentic``) hands an LLM a small
toolbelt and lets the model drive: generate queries, execute them, read the
results, refine, and decide when to stop. This module supplies the shared
machinery for that loop:

- :class:`ToolDef` — one callable tool (name + JSON schema + python fn);
- :class:`ToolRound` — the outcome of ONE provider invocation (assistant
  text + requested tool calls), produced by ``LLM.run_tools``;
- :class:`TraceStep` / :class:`ToolRunResult` — the saved reasoning trace;
- :func:`run_tool_loop` — the provider-dispatched agent loop that executes
  tools, enforces step/wall-clock budgets, and records the trace.

**Neutral message schema.** Conversations are held provider-agnostically as
``list[dict]`` so a run can be serialized and resumed later (with injected
user feedback) on any provider::

    {"role": "user"|"assistant", "content": [block, ...]}

where each block is one of::

    {"type": "text", "text": "..."}
    {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
    {"type": "tool_result", "tool_call_id": "...", "name": "...", "content": "<json str>"}

Providers translate this schema to/from their native format inside
``run_tools`` (Anthropic Messages API tool use; OpenAI Responses API
function tools). The provider SDKs stay lazy imports.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

#: tool results longer than this (JSON chars) are truncated for the model
MAX_TOOL_RESULT_CHARS = 20_000

#: the message injected when a budget is breached, asking for a final answer
WRAP_UP_MESSAGE = (
    "STOP: the search budget is exhausted. Do not issue any more searches. "
    "Call the '{finish_tool}' tool IMMEDIATELY with your best current answer "
    "based on what you have seen so far."
)


@dataclass
class ToolDef:
    """One tool offered to the model.

    ``fn`` receives the parsed arguments dict and may return anything
    JSON-serializable; the return value is serialized (and truncated when
    huge) before being fed back to the model.
    """

    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], Any]


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolRound:
    """The outcome of one provider invocation inside the loop."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TraceStep:
    """One step of the saved reasoning trace. JSON-serializable."""

    index: int
    kind: str  # "assistant_text"|"tool_call"|"tool_result"|"user_feedback"|"system"
    content: str
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolRunResult:
    """The outcome of a full :func:`run_tool_loop` run.

    ``messages`` is the raw conversation in the neutral schema, so a later
    run can resume it (e.g. with injected user feedback appended).
    """

    final_text: str
    steps: list[TraceStep]
    stop_reason: str  # "finish_tool"|"end_turn"|"max_steps"|"budget_exceeded"|"error"
    elapsed_s: float
    usage: dict[str, int]
    messages: list[dict] = field(default_factory=list)


def user_text_message(text: str) -> dict:
    """A neutral-schema user message with one text block."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def serialize_tool_result(value: Any, *, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """JSON-serialize a tool return value, truncating huge payloads sensibly."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = json.dumps({"repr": repr(value)})
    if len(text) > max_chars:
        text = text[:max_chars] + f'... [truncated {len(text) - max_chars} chars]"'
    return text


def _emit(steps: list[TraceStep], on_step: Optional[Callable[[TraceStep], None]],
          kind: str, content: str, *, tool_name: str | None = None,
          arguments: dict | None = None, t0: float = 0.0) -> TraceStep:
    step = TraceStep(index=len(steps), kind=kind, content=content, tool_name=tool_name,
                     arguments=arguments, elapsed_s=round(time.monotonic() - t0, 3))
    steps.append(step)
    if on_step is not None:
        try:
            on_step(step)
        except Exception:  # noqa: BLE001 — a user callback must never kill the loop
            logger.exception("on_step callback raised")
    return step


def run_tool_loop(
    llm,
    *,
    system: str,
    messages: list[dict],
    tools: Sequence[ToolDef],
    max_steps: int = 16,
    budget_seconds: float = 180.0,
    finish_tool: str | None = None,
    on_step: Optional[Callable[[TraceStep], None]] = None,
    max_tokens: int = 4096,
) -> ToolRunResult:
    """Run the provider-dispatched agent loop until the model finishes.

    Args:
        llm: an :class:`patentkit.llm.LLM` whose provider implements
            ``run_tools`` (Anthropic, OpenAI, or a test fake).
        system: the system prompt.
        messages: initial conversation in the neutral schema — typically one
            user message, or a resumed conversation from a previous
            :class:`ToolRunResult` (plus injected feedback messages).
        tools: the toolbelt offered to the model.
        max_steps: maximum provider rounds (model invocations).
        budget_seconds: wall-clock budget for the whole loop.
        finish_tool: name of the tool whose call ends the loop. On a budget
            breach the loop appends one wrap-up user message asking the model
            to call this tool immediately, allows ONE more round, then
            hard-stops.
        on_step: callback invoked for every :class:`TraceStep` as it happens.
        max_tokens: per-round completion budget.

    Returns:
        A :class:`ToolRunResult`. ``stop_reason`` is ``"finish_tool"`` when
        the model called the finish tool within budget, ``"end_turn"`` when
        it stopped calling tools, ``"max_steps"`` / ``"budget_exceeded"``
        when a budget was breached (even if the model then complied during
        the grace round — the run was truncated either way), or ``"error"``
        on a provider failure.
    """
    t0 = time.monotonic()
    tools = list(tools)
    tool_map = {t.name: t for t in tools}
    convo: list[dict] = [dict(m) for m in messages]
    steps: list[TraceStep] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    final_text = ""
    stop_reason = "end_turn"
    breach_reason: str | None = None
    rounds = 0

    while True:
        # --- budget checks before each provider call -----------------------
        over_steps = rounds >= max_steps
        over_budget = (time.monotonic() - t0) >= budget_seconds
        if (over_steps or over_budget) and breach_reason is None:
            breach_reason = "max_steps" if over_steps else "budget_exceeded"
            if finish_tool:
                wrap_up = WRAP_UP_MESSAGE.format(finish_tool=finish_tool)
                convo.append(user_text_message(wrap_up))
                _emit(steps, on_step, "system", wrap_up, t0=t0)
                # fall through: ONE grace round is allowed below
            else:
                stop_reason = breach_reason
                break
        elif (over_steps or over_budget) and breach_reason is not None:
            # the grace round already ran — hard stop
            stop_reason = breach_reason
            break

        # --- one provider round --------------------------------------------
        try:
            round_ = llm.run_tools(convo, system=system, tools=tools, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — surface provider failures in the trace
            logger.exception("tool loop provider call failed")
            _emit(steps, on_step, "system", f"provider error: {exc}", t0=t0)
            stop_reason = "error"
            break
        rounds += 1
        usage["input_tokens"] += round_.input_tokens
        usage["output_tokens"] += round_.output_tokens

        assistant_blocks: list[dict] = []
        if round_.text:
            final_text = round_.text
            assistant_blocks.append({"type": "text", "text": round_.text})
            _emit(steps, on_step, "assistant_text", round_.text, t0=t0)

        if not round_.tool_calls:
            convo.append({"role": "assistant", "content": assistant_blocks or
                          [{"type": "text", "text": round_.text}]})
            stop_reason = breach_reason or "end_turn"
            break

        result_blocks: list[dict] = []
        finished = False
        for call in round_.tool_calls:
            assistant_blocks.append({"type": "tool_call", "id": call.id,
                                     "name": call.name, "arguments": call.arguments})
            _emit(steps, on_step, "tool_call", json.dumps(call.arguments, default=str),
                  tool_name=call.name, arguments=call.arguments, t0=t0)
            tool = tool_map.get(call.name)
            if tool is None:
                payload: Any = {"error": f"Unknown tool {call.name!r}. "
                                         f"Available: {sorted(tool_map)}"}
            else:
                try:
                    payload = tool.fn(call.arguments or {})
                except Exception as exc:  # noqa: BLE001 — report errors to the model
                    logger.exception("tool %s failed", call.name)
                    payload = {"error": f"{call.name} failed: {exc}"}
            content = serialize_tool_result(payload)
            result_blocks.append({"type": "tool_result", "tool_call_id": call.id,
                                  "name": call.name, "content": content})
            _emit(steps, on_step, "tool_result", content, tool_name=call.name, t0=t0)
            if finish_tool and call.name == finish_tool and (
                    not isinstance(payload, dict) or "error" not in payload):
                finished = True

        convo.append({"role": "assistant", "content": assistant_blocks})
        convo.append({"role": "user", "content": result_blocks})

        if finished:
            stop_reason = breach_reason or "finish_tool"
            break

    return ToolRunResult(
        final_text=final_text,
        steps=steps,
        stop_reason=stop_reason,
        elapsed_s=round(time.monotonic() - t0, 3),
        usage=usage,
        messages=convo,
    )


__all__ = [
    "ToolDef",
    "ToolCall",
    "ToolRound",
    "TraceStep",
    "ToolRunResult",
    "run_tool_loop",
    "user_text_message",
    "serialize_tool_result",
    "MAX_TOOL_RESULT_CHARS",
    "WRAP_UP_MESSAGE",
]
