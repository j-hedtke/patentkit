"""MCP stdio server exposing the :class:`PatentToolset`.

One MCP Tool is registered per :data:`~patentkit.integrations.toolset.
TOOL_SPECS` entry (the hand-written JSON schemas pass straight through), and
calls dispatch to a single shared toolset. Run it with the console script::

    patentkit-mcp

Environment configuration:

- ``PATENTKIT_PROVIDER``      "anthropic" or "openai" — LLM for planning/reranking
- ``PATENTKIT_SESSION_DIR``   directory for guided-session JSON persistence
- ``PATENTKIT_INDEX_JSONL``   a .jsonl corpus of Patent records preloaded at startup
- ``USPTO_ODP_API_KEY``       enables file-wrapper context in summarize_key_limitations

Requires the ``mcp`` extra: ``pip install 'patentkit[mcp]'``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from patentkit.integrations.toolset import TOOL_SPECS, PatentToolset, dispatch

logger = logging.getLogger(__name__)


def build_toolset() -> PatentToolset:
    """Build the server's toolset from PATENTKIT_* environment variables."""
    toolset = PatentToolset(
        provider=os.environ.get("PATENTKIT_PROVIDER"),
        session_dir=os.environ.get("PATENTKIT_SESSION_DIR"),
    )
    corpus = os.environ.get("PATENTKIT_INDEX_JSONL")
    if corpus:
        outcome = toolset.index_patents(jsonl_path=corpus)
        logger.info("Preloaded corpus from %s: %s", corpus, outcome)
    return toolset


def _import_mcp():
    try:
        import mcp.types as types  # noqa: PLC0415 — optional extra
        from mcp.server import Server  # noqa: PLC0415
        from mcp.server.stdio import stdio_server  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The patentkit MCP server requires the 'mcp' package. "
            "Install it with: pip install 'patentkit[mcp]'"
        ) from exc
    return types, Server, stdio_server


def build_server(toolset: PatentToolset):
    """Create the low-level MCP Server with one Tool per TOOL_SPECS entry."""
    types, Server, _ = _import_mcp()
    server = Server("patentkit")

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["parameters"],
            )
            for spec in TOOL_SPECS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[Any]:
        result = dispatch(toolset, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


async def _serve() -> None:
    _, _, stdio_server = _import_mcp()
    toolset = build_toolset()
    server = build_server(toolset)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Console entry point (``patentkit-mcp``): run the stdio MCP server."""
    import asyncio

    logging.basicConfig(level=os.environ.get("PATENTKIT_LOG_LEVEL", "INFO"))
    try:
        asyncio.run(_serve())
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
