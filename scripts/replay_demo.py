"""Replay a captured demo log with readable pacing (for recording the README
GIF with vhs — see docs/assets/demo.tape). The content is the real captured
output of examples/demo_invalidity_chart.py; only the timing is synthetic.

Usage: python scripts/replay_demo.py [logfile] [width]
"""

from __future__ import annotations

import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "data/demo_run.log"
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 100

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def style(line: str) -> tuple[str, float]:
    """Return (styled line, delay after printing it)."""
    if line.startswith("━━━"):
        return f"\n{BOLD}{CYAN}{line}{RESET}", 1.2
    if line.startswith("agent -> "):
        return f"{GREEN}{line}{RESET}", 0.25
    if line.startswith("assistant_text:"):
        return f"{BOLD}{line}{RESET}", 0.6
    if "◀ ground truth" in line:
        return f"{BOLD}{GREEN}{line}{RESET}", 0.8
    if line.startswith(("search_patents ->", "get_patent ->")):
        return f"{DIM}{line}{RESET}", 0.12
    return line, 0.25


def main() -> None:
    print(f"{BOLD}${RESET} python examples/demo_invalidity_chart.py", flush=True)
    time.sleep(1.0)
    for raw in open(LOG):
        line = raw.rstrip("\n")
        if len(line) > WIDTH:
            line = line[: WIDTH - 1] + "…"
        styled, delay = style(line)
        print(styled, flush=True)
        time.sleep(delay)


if __name__ == "__main__":
    main()
