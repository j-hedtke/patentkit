"""Agentic freedom-to-operate (FTO) search.

With an LLM configured, an agent conversation (:class:`~patentkit.agents.
agentic.AgenticSearchRunner`) drives the screen: the model generates queries
from the product description, executes them, inspects hits, and finishes
with a ranked candidate list — with a full saved trace and a resumable
conversation. Without an LLM the agent degrades to a SINGLE plain BM25 pass
(clearly labeled ``mode="degraded_keyword_only"``), so it works keys-free.

**In-force approximation**: when ``in_force_only`` is set, the search is
biased to patents filed within the last :data:`IN_FORCE_YEARS` years. US
utility patents expire 20 years from the earliest non-provisional filing
date; the extra year absorbs provisional priority and patent-term
adjustment. This is a recall filter only — actual term/fee status must be
verified per patent.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.agents._support import report_progress, result_to_dict
from patentkit.agents.agentic import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_STEPS,
    AgenticSearchRunner,
    SearchTrace,
)
from patentkit.agents.invalidity_search import candidates_to_result_dicts, step_summary
from patentkit.agents.planner import _fallback_keywords
from patentkit.llm.tools import TraceStep
from patentkit.models import PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery, VectorStore

logger = logging.getLogger(__name__)

#: years back from today used to approximate "possibly still in force"
IN_FORCE_YEARS = 21


class FtoSearchResult(BaseModel):
    """Serializable outcome of an FTO search.

    .. warning::
        FTO conclusions are legal opinions. ``requires_attorney_review`` is
        always ``True``: these results identify patents *worth reviewing*,
        they are not a clearance opinion. Claim scope, term, fee status, and
        prosecution history must be assessed by a qualified attorney.
    """

    product_description: str
    jurisdiction: str = "US"
    in_force_only: bool = True
    plan_or_params: dict = Field(default_factory=dict)
    #: ranked patents, best first:
    #: {patent_number, title, score (=confidence 0-1), passages, why}
    results: list[dict] = Field(default_factory=list)
    excluded: dict[str, list[str]] = Field(default_factory=dict)
    timing: dict[str, float] = Field(default_factory=dict)
    trace: Optional[SearchTrace] = None
    stop_reason: Optional[str] = None
    #: neutral-schema agent conversation for resumption (agentic mode)
    conversation: Optional[list[dict]] = None
    requires_attorney_review: bool = True


class FtoSearchAgent:
    """Agentic FTO screening over a patent store (keys-free degraded mode
    falls back to a single BM25 pass)."""

    def __init__(self, keyword_store: KeywordStore, vector_store: Optional[VectorStore] = None,
                 llm=None):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.llm = llm

    def search(
        self,
        product_description: str,
        *,
        jurisdiction: str = "US",
        in_force_only: bool = True,
        final_k: int = 25,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Optional[Callable[[TraceStep], None]] = None,
        custom_exclusions: Sequence[str] = (),
        feedback_messages: Sequence[str] = (),
        resume_messages: Optional[list[dict]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> FtoSearchResult:
        """Screen the corpus for patents the product might practice.

        Args:
            product_description: free-text description of the product/feature.
            jurisdiction: country code filter hint (default "US").
            in_force_only: bias to patents filed within the last
                :data:`IN_FORCE_YEARS` years (documented approximation above).
            final_k: ranked results returned.
            budget_seconds / max_steps: agent budgets (agentic mode).
            on_step: live callback receiving every trace step.
            custom_exclusions: patent numbers to exclude (enforced in the
                tool layer).
            feedback_messages / resume_messages: continue a previous agent
                conversation with injected user feedback.
            progress: optional callback receiving human-readable updates.
        """
        t0 = time.monotonic()
        window_start = (date.today() - timedelta(days=int(IN_FORCE_YEARS * 365.25))
                        if in_force_only else None)
        excluded: dict[str, list[str]] = {}
        if custom_exclusions:
            excluded["custom"] = sorted({str(n) for n in custom_exclusions})

        if self.llm is None:
            return self._degraded_search(product_description, jurisdiction, in_force_only,
                                         window_start, excluded, final_k,
                                         progress=progress, t0=t0)

        def _step(step: TraceStep) -> None:
            report_progress(progress, step_summary(step))
            if on_step is not None:
                on_step(step)

        runner = AgenticSearchRunner(self.keyword_store, self.vector_store, self.llm,
                                     max_steps=max_steps, budget_seconds=budget_seconds)
        description = product_description
        if jurisdiction:
            description += f"\nJurisdiction of interest: {jurisdiction}"
        outcome = runner.run(
            "fto",
            product_description=description,
            exclusions=excluded,
            cutoff_date=window_start,
            final_k=final_k,
            feedback_messages=feedback_messages,
            resume_messages=resume_messages,
            on_step=_step,
        )
        return FtoSearchResult(
            product_description=product_description,
            jurisdiction=jurisdiction,
            in_force_only=in_force_only,
            plan_or_params={
                "mode": "agentic",
                "after_date": window_start.isoformat() if window_start else None,
                "countries": [jurisdiction] if jurisdiction else [],
                "final_k": final_k,
                "max_steps": max_steps,
                "budget_seconds": budget_seconds,
                "queries_issued": len(outcome.trace.queries),
                "rationale": outcome.rationale,
                "suggested_next_queries": outcome.suggested_next_queries,
            },
            results=candidates_to_result_dicts(outcome.results),
            excluded=excluded,
            timing={"total": round(time.monotonic() - t0, 4), "agent": outcome.elapsed_s},
            trace=outcome.trace,
            stop_reason=outcome.stop_reason,
            conversation=outcome.messages,
        )

    # ------------------------------------------------------- degraded mode
    def _degraded_search(self, product_description: str, jurisdiction: str,
                         in_force_only: bool, window_start: Optional[date],
                         excluded: dict[str, list[str]], final_k: int, *,
                         progress: Optional[Callable[[str], None]],
                         t0: float) -> FtoSearchResult:
        """Keys-free fallback: ONE plain BM25 pass with the in-force window."""
        keywords = _fallback_keywords(product_description=product_description)
        exclude_numbers: list[PatentNumber] = []
        for numbers in excluded.values():
            for raw in numbers:
                try:
                    exclude_numbers.append(PatentNumber.parse(raw))
                except ValueError:
                    logger.warning("Skipping unparseable exclusion number: %r", raw)
        query = SearchQuery(
            keywords=keywords,
            text=product_description,
            after_date=window_start,
            countries=[jurisdiction] if jurisdiction else [],
            exclude_numbers=exclude_numbers,
            limit=max(final_k, 25),
        )
        report_progress(progress, "degraded keys-free mode: single keyword pass "
                                  f"({len(keywords)} keywords, after {window_start})")
        hits = self.keyword_store.search(query)
        max_score = max((r.score for r in hits), default=0.0) or 1.0
        results = [result_to_dict(r, r.score / max_score,
                                  "degraded mode: BM25 keyword relevance only")
                   for r in hits[:final_k]]
        return FtoSearchResult(
            product_description=product_description,
            jurisdiction=jurisdiction,
            in_force_only=in_force_only,
            plan_or_params={
                "mode": "degraded_keyword_only",
                "keywords": keywords,
                "after_date": window_start.isoformat() if window_start else None,
                "countries": [jurisdiction] if jurisdiction else [],
                "final_k": final_k,
            },
            results=results,
            excluded=excluded,
            timing={"total": round(time.monotonic() - t0, 4)},
            trace=None,
            stop_reason="degraded",
        )


__all__ = ["FtoSearchAgent", "FtoSearchResult", "IN_FORCE_YEARS"]
