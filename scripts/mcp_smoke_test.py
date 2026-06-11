"""Smoke-test the patentkit MCP server over stdio, exactly as Claude Desktop
drives it: list tools, run a search, build a markdown claim chart, chart one
limitation across references, and export the chart to DOCX.

Usage:
    python scripts/mcp_smoke_test.py [--patent US10491679B2] \
        [--references US20060235700A1 US6438545B1] [--limitation "..."]

Env: same as the Claude Desktop config — ANTHROPIC_API_KEY,
PATENTKIT_INDEX_JSONL, PATENTKIT_SESSION_DIR are forwarded to the server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patent", default="US10491679B2")
    parser.add_argument("--claim", type=int, default=1)
    parser.add_argument("--references", nargs="+",
                        default=["US20060235700A1", "US6438545B1"])
    parser.add_argument("--limitation", default="receiving")
    args = parser.parse_args()

    server = StdioServerParameters(
        command=".venv/bin/patentkit-mcp",
        env={k: v for k, v in os.environ.items()
             if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PATENTKIT_INDEX_JSONL",
                      "PATENTKIT_SESSION_DIR", "PATH", "HOME")},
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"[1] tools ({len(names)}): {', '.join(names)}\n")

            result = await session.call_tool(
                "search_patents", {"keywords": ["voice command", "mobile device",
                                                "remote computer control"], "limit": 5})
            print(f"[2] search_patents →\n{text_of(result)[:600]}\n")

            result = await session.call_tool("build_claim_chart", {
                "patent_number": args.patent, "claim_number": args.claim,
                "reference_numbers": args.references[:1],
            })
            payload = json.loads(text_of(result))
            md = payload.get("markdown", "")
            print(f"[3] build_claim_chart markdown ({len(md)} chars):\n{md[:1500]}\n")
            assert md.startswith("##"), "markdown chart missing/odd format"

            result = await session.call_tool("chart_limitation", {
                "limitation": args.limitation, "patent": args.patent,
                "claim_number": args.claim, "references": args.references,
            })
            payload = json.loads(text_of(result))
            md = payload.get("markdown", "")
            print(f"[4] chart_limitation markdown:\n{md[:1200]}\n")

            result = await session.call_tool("export_claim_chart_docx", {
                "patent": args.patent, "claim_number": args.claim,
            })
            print(f"[5] export_claim_chart_docx →\n{text_of(result)[:300]}\n")

            print("SMOKE TEST OK")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
