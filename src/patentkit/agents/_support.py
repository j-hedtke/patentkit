"""Private helpers shared by the agents.

The staged brute-force pipeline helpers (RRF fusion, query merging, batched
LLM scoring, 0.75/0.25 score combining) were removed when the search agents
moved to the pure agentic core (``patentkit.agents.agentic``). Only the
small, still-reused utilities live here.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from patentkit.search.base import SearchResult

logger = logging.getLogger(__name__)


def report_progress(progress: Optional[Callable[[str], None]], message: str) -> None:
    """Invoke a user progress callback, never letting it break the pipeline."""
    logger.info(message)
    if progress is not None:
        try:
            progress(message)
        except Exception:  # noqa: BLE001 — user callback must not kill a search
            logger.exception("progress callback raised")


def result_to_dict(result: SearchResult, score: float, why: Optional[str]) -> dict:
    """Serialize one ranked keyword-store result for the result models."""
    return {
        "patent_number": str(result.patent_number),
        "title": result.patent.title if result.patent else None,
        "score": round(float(score), 4),
        "passages": [
            {"text": p.text, "field": p.field, "score": round(float(p.score), 4)}
            for p in result.passages
        ],
        "why": why,
    }
