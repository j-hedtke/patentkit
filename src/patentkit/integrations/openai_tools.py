"""OpenAI function-calling adapter over the :class:`PatentToolset`.

- :func:`openai_tool_definitions` converts :data:`TOOL_SPECS` into the
  ``{"type": "function", "function": {...}}`` shape Chat Completions and the
  Responses API accept;
- :func:`handle_tool_call` executes one tool call and returns the JSON
  string to feed back to the model;
- :func:`run_agent_loop` is a minimal example harness around the Responses
  API (requires ``pip install 'patentkit[openai]'``).
"""

from __future__ import annotations

import json
import logging

from patentkit.integrations.toolset import TOOL_SPECS, PatentToolset, dispatch

logger = logging.getLogger(__name__)


def openai_tool_definitions() -> list[dict]:
    """TOOL_SPECS in OpenAI function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in TOOL_SPECS
    ]


def handle_tool_call(toolset: PatentToolset, name: str, arguments_json: str) -> str:
    """Execute one model-emitted tool call; returns the tool output as JSON.

    Malformed argument JSON is reported back to the model as an error dict
    rather than raising, so agent loops can self-correct.
    """
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Arguments were not valid JSON: {exc}"})
    return json.dumps(dispatch(toolset, name, arguments), default=str)


def run_agent_loop(prompt: str, toolset: PatentToolset, model: str = "gpt-5.1",
                   max_turns: int = 10) -> str:
    """Minimal Responses-API tool loop — an example harness, not a framework.

    Sends ``prompt`` with the patentkit tools attached, executes every
    function call the model makes, feeds outputs back, and returns the
    model's final text. Requires the ``openai`` package and an
    ``OPENAI_API_KEY``.

    Example::

        from patentkit.integrations import PatentToolset, run_agent_loop
        toolset = PatentToolset()
        toolset.index_patents(jsonl_path="corpus.jsonl")
        print(run_agent_loop("Find prior art for US10123456B2 claim 1", toolset))
    """
    try:
        import openai  # noqa: PLC0415 — optional extra
    except ImportError as exc:
        raise ImportError(
            "run_agent_loop requires the openai package: pip install 'patentkit[openai]'"
        ) from exc

    client = openai.OpenAI()
    tools = [
        {"type": "function", "name": spec["name"], "description": spec["description"],
         "parameters": spec["parameters"]}
        for spec in TOOL_SPECS
    ]
    input_items: list[dict] = [{"role": "user", "content": prompt}]

    for turn in range(max_turns):
        response = client.responses.create(model=model, input=input_items, tools=tools)
        calls = [item for item in response.output if getattr(item, "type", "") == "function_call"]
        if not calls:
            return response.output_text
        input_items += [item.model_dump() for item in response.output]
        for call in calls:
            logger.info("turn %d: tool call %s", turn, call.name)
            output = handle_tool_call(toolset, call.name, call.arguments)
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })
    return "Agent loop ended without a final answer (max_turns reached)."


__all__ = ["openai_tool_definitions", "handle_tool_call", "run_agent_loop"]
