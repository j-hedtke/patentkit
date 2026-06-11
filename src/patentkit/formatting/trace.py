"""Render agentic :class:`~patentkit.agents.agentic.SearchTrace` objects as a
readable markdown narrative.

The renderer is duck-typed (it only touches trace attributes), dependency-free,
and deliberately compact: thinking text and tool payloads are capped so the
output pastes cleanly into a chat client and renders as a legible reasoning
trace — one section per agent round with the round's thinking, the queries it
issued (inline code), result counts, and shortlist updates.
"""

from __future__ import annotations

import json
from typing import Any, Optional

__all__ = ["search_trace_markdown"]

#: cap on rendered assistant thinking text per round
_TEXT_CAP = 700
#: cap on rendered tool-call argument JSON
_ARGS_CAP = 220
#: cap on miscellaneous result snippets
_SNIPPET_CAP = 160


def _cap(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + " …"
    return text


def _args_code(arguments: Optional[dict]) -> str:
    if not arguments:
        return ""
    return " " + _cap(json.dumps(arguments, default=str), _ARGS_CAP)


def _result_summary(tool_name: Optional[str], content: str) -> str:
    """One short phrase describing a tool result (counts, not payloads)."""
    try:
        payload: Any = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _cap(content, _SNIPPET_CAP)
    if not isinstance(payload, dict):
        return _cap(str(payload), _SNIPPET_CAP)
    if "error" in payload:
        return f"error: {_cap(str(payload['error']), _SNIPPET_CAP)}"
    if "count" in payload:  # search tools
        numbers = [str(r.get("number", "?")) for r in (payload.get("results") or [])[:5]
                   if isinstance(r, dict)]
        suffix = f" (top: {', '.join(numbers)})" if numbers else ""
        return f"{payload['count']} result(s){suffix}"
    if "accepted" in payload:  # shortlist / finish
        parts = [f"{payload['accepted']} candidate(s) accepted"]
        rejected = payload.get("rejected") or []
        if rejected:
            parts.append(f"{len(rejected)} rejected")
        return ", ".join(parts)
    if "title" in payload or "number" in payload:  # get_patent
        title = payload.get("title") or ""
        return _cap(f"{payload.get('number', '')} — {title}".strip(" —"), _SNIPPET_CAP)
    return _cap(content, _SNIPPET_CAP)


def _group_rounds(steps) -> list[dict]:
    """Group trace steps into agent rounds.

    A new round starts at each assistant-text step (one provider invocation
    emits its thinking first); tool calls/results without preceding text
    attach to the current round.
    """
    rounds: list[dict] = []
    current: Optional[dict] = None

    def new_round() -> dict:
        round_: dict = {"texts": [], "events": []}
        rounds.append(round_)
        return round_

    for step in steps:
        if step.kind == "assistant_text":
            current = new_round()
            current["texts"].append(step.content)
            continue
        if current is None:
            current = new_round()
        if step.kind == "system":
            current["events"].append(("note", None, step.content, None))
        elif step.kind == "tool_call":
            current["events"].append(("call", step.tool_name, step.content, step.arguments))
        elif step.kind == "tool_result":
            current["events"].append(("result", step.tool_name, step.content, None))
        else:  # user_feedback or future kinds
            current["events"].append(("note", step.tool_name, step.content, None))
    return rounds


def _render_round_events(events: list[tuple]) -> list[str]:
    """Pair each tool call with its result into one compact bullet."""
    lines: list[str] = []
    pending: Optional[tuple[str, Optional[dict]]] = None  # (tool_name, arguments)

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            name, arguments = pending
            lines.append(f"- `{name}{_args_code(arguments)}`")
            pending = None

    for kind, tool_name, content, arguments in events:
        if kind == "call":
            flush_pending()
            pending = (tool_name or "?", arguments)
        elif kind == "result":
            if pending is not None and pending[0] == tool_name:
                name, args = pending
                lines.append(f"- `{name}{_args_code(args)}` → {_result_summary(tool_name, content)}")
                pending = None
            else:
                flush_pending()
                lines.append(f"- `{tool_name}` → {_result_summary(tool_name, content)}")
        else:  # note (budget wrap-ups, provider errors, injected events)
            flush_pending()
            lines.append(f"> {_cap(content, _TEXT_CAP)}")
    flush_pending()
    return lines


def search_trace_markdown(trace) -> str:
    """Render a :class:`SearchTrace` as a chat-ready markdown narrative.

    One section per agent round: the agent's thinking text, each query it
    issued as inline code with its result count, shortlist updates, plus the
    injected user feedback and the stop reason. Quotes and payloads are
    capped so the whole trace stays pasteable.
    """
    queries = list(getattr(trace, "queries", []) or [])
    shortlist_history = list(getattr(trace, "shortlist_history", []) or [])
    lines = [
        f"# Reasoning trace — agentic {trace.search_type} search",
        "",
        f"**Target:** {trace.target or '(none)'}  ",
        f"**Budgets:** max_steps={trace.max_steps}, budget_seconds={trace.budget_seconds:g}  ",
        f"**Stop reason:** `{trace.stop_reason or '(running)'}` after {trace.elapsed_s:.1f}s — "
        f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} issued, "
        f"{len(shortlist_history)} shortlist revision(s)",
        "",
    ]
    feedback = list(getattr(trace, "feedback", []) or [])
    if feedback:
        lines.append("## Injected user feedback")
        lines += [f"- {_cap(str(f), _TEXT_CAP)}" for f in feedback]
        lines.append("")

    for i, round_ in enumerate(_group_rounds(trace.steps), start=1):
        lines.append(f"## Round {i}")
        for text in round_["texts"]:
            lines.append(_cap(text, _TEXT_CAP))
        event_lines = _render_round_events(round_["events"])
        if event_lines:
            if round_["texts"]:
                lines.append("")
            lines += event_lines
        lines.append("")

    if shortlist_history:
        lines.append("## Shortlist evolution")
        for i, snapshot in enumerate(shortlist_history, start=1):
            numbers = ", ".join(str(c.get("number", "?")) for c in snapshot) or "(empty)"
            lines.append(f"- revision {i} ({len(snapshot)}): {numbers}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
