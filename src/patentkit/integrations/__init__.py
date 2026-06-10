"""Exposure layers: the shared :class:`PatentToolset`, the MCP stdio server,
the OpenAI function-calling adapter, and the Claude Code plugin (see
``plugins/claude/`` in the repository)."""

from patentkit.integrations.openai_tools import (
    handle_tool_call,
    openai_tool_definitions,
    run_agent_loop,
)
from patentkit.integrations.toolset import TOOL_SPECS, PatentToolset, dispatch

__all__ = [
    "PatentToolset",
    "TOOL_SPECS",
    "dispatch",
    "openai_tool_definitions",
    "handle_tool_call",
    "run_agent_loop",
]
