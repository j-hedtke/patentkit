"""Tests for the shared provider-agnostic tool loop (llm/tools.py)."""

from __future__ import annotations

import json

import pytest

from patentkit.llm.base import LLM
from patentkit.llm.routing import ModelChoice
from patentkit.llm.tools import (
    MAX_TOOL_RESULT_CHARS,
    ToolDef,
    TraceStep,
    run_tool_loop,
    serialize_tool_result,
    user_text_message,
)
from tests.fakes import FakeLLM


# ----------------------------------------------------------------- helpers

def make_tools(record: list | None = None, finishes: list | None = None) -> list[ToolDef]:
    record = record if record is not None else []
    finishes = finishes if finishes is not None else []

    def echo(args: dict):
        record.append(args)
        return {"echoed": args}

    def boom(args: dict):
        raise RuntimeError("kaboom")

    def finish(args: dict):
        finishes.append(args)
        return {"ok": True}

    schema = {"type": "object", "properties": {}, "required": []}
    return [
        ToolDef("echo", "echo arguments back", schema, echo),
        ToolDef("boom", "always raises", schema, boom),
        ToolDef("finish", "final answer", schema, finish),
    ]


def run(llm: FakeLLM, tools: list[ToolDef], **kwargs):
    return run_tool_loop(
        llm, system="You are a test agent.",
        messages=[user_text_message("do the thing")],
        tools=tools, finish_tool="finish", **kwargs,
    )


# ------------------------------------------------------------------- basics

def test_finish_tool_ends_loop_with_ordered_trace() -> None:
    record: list = []
    finishes: list = []
    llm = FakeLLM(tool_script=[
        {"text": "let me try a query",
         "tool_calls": [{"name": "echo", "arguments": {"q": "alpha"}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"answer": 42}}]},
        {"text": "should never be reached"},
    ])
    result = run(llm, make_tools(record, finishes))

    assert result.stop_reason == "finish_tool"
    assert record == [{"q": "alpha"}]
    assert finishes == [{"answer": 42}]
    assert result.final_text == "let me try a query"
    assert result.usage == {"input_tokens": 2, "output_tokens": 2}
    kinds = [s.kind for s in result.steps]
    assert kinds == ["assistant_text", "tool_call", "tool_result", "tool_call", "tool_result"]
    assert [s.index for s in result.steps] == list(range(5))
    assert all(s.elapsed_s >= 0 for s in result.steps)
    # tool steps carry the tool name and (for calls) the arguments
    assert result.steps[1].tool_name == "echo"
    assert result.steps[1].arguments == {"q": "alpha"}
    # every step is JSON-serializable
    json.dumps([s.to_dict() for s in result.steps])


def test_end_turn_when_model_stops_calling_tools() -> None:
    llm = FakeLLM(tool_script=[{"text": "all done, no tools needed"}])
    result = run(llm, make_tools())
    assert result.stop_reason == "end_turn"
    assert result.final_text == "all done, no tools needed"
    assert result.messages[-1]["role"] == "assistant"


def test_system_prompt_and_conversation_reach_the_provider() -> None:
    llm = FakeLLM(tool_script=[{"text": "ok"}])
    run(llm, make_tools())
    assert llm.tool_systems == ["You are a test agent."]
    first = llm.tool_conversations[0][0]
    assert first["role"] == "user"
    assert first["content"][0]["text"] == "do the thing"


def test_tool_error_is_reported_to_model_and_loop_continues() -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "boom", "arguments": {}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"answer": "recovered"}}]},
    ])
    result = run(llm, make_tools())
    assert result.stop_reason == "finish_tool"
    boom_result = next(s for s in result.steps if s.kind == "tool_result" and s.tool_name == "boom")
    assert "kaboom" in boom_result.content


def test_unknown_tool_returns_error_result() -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
        {"tool_calls": [{"name": "finish", "arguments": {}}]},
    ])
    result = run(llm, make_tools())
    unknown = next(s for s in result.steps if s.tool_name == "no_such_tool" and s.kind == "tool_result")
    assert "Unknown tool" in unknown.content
    assert result.stop_reason == "finish_tool"


def test_finish_tool_error_does_not_end_loop() -> None:
    """A finish call rejected by the tool layer (error payload) keeps going."""
    attempts: list = []

    def finicky_finish(args: dict):
        attempts.append(args)
        if len(attempts) == 1:
            return {"error": "candidates required"}
        return {"ok": True}

    schema = {"type": "object", "properties": {}, "required": []}
    tools = [ToolDef("finish", "final answer", schema, finicky_finish)]
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "finish", "arguments": {}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"candidates": []}}]},
    ])
    result = run(llm, tools)
    assert len(attempts) == 2
    assert result.stop_reason == "finish_tool"


# ------------------------------------------------------------------ budgets

def test_max_steps_injects_wrap_up_and_allows_one_grace_round() -> None:
    record: list = []
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "echo", "arguments": {"round": 1}}]},
        {"tool_calls": [{"name": "echo", "arguments": {"round": 2}}]},
        {"tool_calls": [{"name": "echo", "arguments": {"round": 3}}]},  # grace round, ignores ask
        {"tool_calls": [{"name": "echo", "arguments": {"round": 4}}]},  # never reached
    ])
    result = run(llm, make_tools(record), max_steps=2)

    assert result.stop_reason == "max_steps"
    # exactly one grace round beyond max_steps ran
    assert [a["round"] for a in record] == [1, 2, 3]
    # the wrap-up message was injected into the trace and the conversation
    system_steps = [s for s in result.steps if s.kind == "system"]
    assert len(system_steps) == 1 and "finish" in system_steps[0].content
    wrap_up_texts = [
        block["text"]
        for message in result.messages if message["role"] == "user"
        for block in message["content"] if block.get("type") == "text"
    ]
    assert any("budget is exhausted" in t for t in wrap_up_texts)


def test_max_steps_grace_round_compliance_still_reports_truncation() -> None:
    finishes: list = []
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "echo", "arguments": {}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"answer": "best effort"}}]},
    ])
    result = run(llm, make_tools(finishes=finishes), max_steps=1)
    assert result.stop_reason == "max_steps"  # the run was truncated
    assert finishes == [{"answer": "best effort"}]  # but the answer was captured


def test_budget_exceeded_injects_wrap_up_message() -> None:
    finishes: list = []
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "finish", "arguments": {"answer": "rushed"}}]},
    ])
    result = run(llm, make_tools(finishes=finishes), budget_seconds=0.0)
    assert result.stop_reason == "budget_exceeded"
    assert finishes == [{"answer": "rushed"}]
    assert result.steps[0].kind == "system"
    assert "budget is exhausted" in result.steps[0].content


def test_budget_without_finish_tool_stops_immediately() -> None:
    llm = FakeLLM(tool_script=[{"text": "never called"}])
    result = run_tool_loop(llm, system="s", messages=[user_text_message("go")],
                           tools=make_tools(), budget_seconds=0.0, finish_tool=None)
    assert result.stop_reason == "budget_exceeded"
    assert llm.tool_conversations == []  # no provider round happened


def test_provider_error_stops_with_error_reason() -> None:
    class ExplodingLLM(LLM):
        def __init__(self):
            super().__init__(ModelChoice("fake", "fake-model"))

        def run_tools(self, messages, *, system=None, tools=(), max_tokens=4096):
            raise RuntimeError("provider down")

    result = run_tool_loop(ExplodingLLM(), system="s",
                           messages=[user_text_message("go")], tools=make_tools(),
                           finish_tool="finish")
    assert result.stop_reason == "error"
    assert any("provider down" in s.content for s in result.steps)


def test_base_llm_run_tools_raises_helpfully() -> None:
    llm = LLM(ModelChoice("fake", "fake-model"))
    with pytest.raises(NotImplementedError, match="does not support tool use"):
        llm.run_tools([], tools=[])


# --------------------------------------------------------------- resumption

def test_messages_round_trip_resumes_conversation() -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "echo", "arguments": {"q": "first run"}}]},
        {"text": "pausing here"},
        # rounds for the resumed run:
        {"tool_calls": [{"name": "finish", "arguments": {"answer": "resumed"}}]},
    ])
    tools = make_tools()
    first = run(llm, tools)
    assert first.stop_reason == "end_turn"

    # serialize / restore the conversation, inject feedback, and resume
    restored = json.loads(json.dumps(first.messages))
    resumed_input = restored + [user_text_message("user feedback: dig deeper")]
    second = run_tool_loop(llm, system="You are a test agent.", messages=resumed_input,
                           tools=tools, finish_tool="finish")
    assert second.stop_reason == "finish_tool"
    # the resumed provider call saw the full first-run history + the feedback
    seen = llm.tool_conversations[-1]
    texts = [b.get("text", "") for m in seen for b in m["content"] if isinstance(b, dict)]
    assert any("do the thing" in t for t in texts)
    assert any("dig deeper" in t for t in texts)
    # and the prior tool exchange survived the round trip
    assert any(b.get("type") == "tool_result" for m in seen for b in m["content"])
    # second.messages extends the resumed history
    assert second.messages[: len(restored)] == restored


# --------------------------------------------------------------- serializing

def test_serialize_tool_result_truncates_huge_payloads() -> None:
    huge = {"blob": "x" * (2 * MAX_TOOL_RESULT_CHARS)}
    text = serialize_tool_result(huge)
    assert len(text) < 2 * MAX_TOOL_RESULT_CHARS
    assert "truncated" in text


def test_serialize_tool_result_handles_unserializable_objects() -> None:
    class Weird:
        pass

    text = serialize_tool_result({"w": Weird()})
    json.loads(text)  # still valid JSON


def test_on_step_callback_failures_do_not_break_the_loop() -> None:
    def bad_callback(step: TraceStep) -> None:
        raise RuntimeError("callback bug")

    llm = FakeLLM(tool_script=[{"tool_calls": [{"name": "finish", "arguments": {}}]}])
    result = run_tool_loop(llm, system="s", messages=[user_text_message("go")],
                           tools=make_tools(), finish_tool="finish", on_step=bad_callback)
    assert result.stop_reason == "finish_tool"
