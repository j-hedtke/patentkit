"""Agentic invalidity (prior-art) search.

With an LLM configured, the search is run by ONE agent conversation on the
provider's native tool-use platform (:class:`~patentkit.agents.agentic.
AgenticSearchRunner`): the model generates keyword queries itself, executes
them as tools, reads the results, refines its angles, and decides when to
stop — under explicit step/wall-clock budgets, with a full saved reasoning
trace and resumable conversation for user-feedback injection.

The tool layer (not the prompt) enforces the invalidity ground rules: the
prior-art cutoff (``before_date`` clamped to the target's earliest
priority/filing date) and the default exclusions (examiner-cited art — with
best-effort file-wrapper enrichment, family members, the target itself, and
any custom numbers).

Without an LLM the agent degrades to a SINGLE plain BM25 keyword pass with
the same exclusions — clearly labeled ``mode="degraded_keyword_only"`` in
the result — so the toolkit keeps working keys-free.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.agents._support import report_progress, result_to_dict
from patentkit.agents.agentic import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_STEPS,
    AgenticCandidate,
    AgenticSearchOutcome,
    AgenticSearchRunner,
    SearchTrace,
)
from patentkit.agents.planner import _fallback_keywords
from patentkit.llm.tools import TraceStep
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery, VectorStore

logger = logging.getLogger(__name__)


class InvaliditySearchResult(BaseModel):
    """Serializable outcome of one invalidity search run."""

    target: str
    claims: list[int] = Field(default_factory=list)
    #: the run parameters actually applied (cutoff, budgets, mode)
    plan_or_params: dict = Field(default_factory=dict)
    #: ranked references, best first:
    #: {patent_number, title, score (=confidence 0-1), passages, why}
    results: list[dict] = Field(default_factory=list)
    #: exclusion reason -> patent numbers excluded for that reason
    excluded: dict[str, list[str]] = Field(default_factory=dict)
    timing: dict[str, float] = Field(default_factory=dict)
    #: full reasoning trace (agentic mode) — queries issued, tool results,
    #: shortlist evolution; None in degraded keys-free mode
    trace: Optional[SearchTrace] = None
    #: why the agent stopped: finish_tool | end_turn | max_steps |
    #: budget_exceeded | error | degraded
    stop_reason: Optional[str] = None
    #: neutral-schema agent conversation; pass back as ``resume_messages``
    #: (with ``feedback_messages``) to continue the SAME agent run
    conversation: Optional[list[dict]] = None


def candidates_to_result_dicts(candidates: list[AgenticCandidate]) -> list[dict]:
    """Map agent candidates to the public result-dict shape, best first."""
    return [
        {
            "patent_number": c.number,
            "title": c.title,
            "score": round(c.confidence, 4),
            "passages": [{"text": p, "field": "agent", "score": round(c.confidence, 4)}
                         for p in c.passages],
            "why": c.why or None,
        }
        for c in candidates
    ]


def step_summary(step: TraceStep) -> str:
    """One-line human-readable rendering of a trace step (for progress UIs)."""
    if step.kind == "tool_call":
        return f"agent -> {step.tool_name}: {step.content[:160]}"
    if step.kind == "tool_result":
        return f"{step.tool_name} -> {step.content[:160]}"
    return f"{step.kind}: {step.content[:160]}"


class InvaliditySearchAgent:
    """Pure agentic prior-art search over pluggable stores.

    Args:
        keyword_store: any :class:`~patentkit.search.base.KeywordStore`.
        vector_store: optional :class:`~patentkit.search.base.VectorStore`;
            registers the agent's ``semantic_search`` tool when present.
        llm: optional :class:`patentkit.llm.LLM` driving the agent; ``None``
            runs the keys-free degraded single-pass keyword search.
        file_wrapper: optional client with an ``enrich_patent(patent)``
            method used to recover examiner-cited art missing from the
            patent record; failures are logged and skipped.
    """

    def __init__(
        self,
        keyword_store: KeywordStore,
        vector_store: Optional[VectorStore] = None,
        llm=None,
        file_wrapper=None,
    ):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.llm = llm
        self.file_wrapper = file_wrapper

    # ----------------------------------------------------------- exclusions
    def _build_exclusions(
        self,
        patent: Patent,
        exclude_examiner_art: bool,
        exclude_family: bool,
        custom_exclusions: Sequence[str],
    ) -> dict[str, list[str]]:
        """Default exclusion sets, keyed by reason; enforced at the tool layer."""
        excluded: dict[str, list[str]] = {"self": [str(patent.patent_number)]}
        if exclude_examiner_art:
            examiner = set(patent.examiner_cited_numbers)
            enriched = self._enriched_examiner_numbers(patent)
            excluded["examiner_cited"] = sorted(examiner | enriched)
        if exclude_family:
            excluded["family"] = sorted({str(n) for n in patent.family})
        if custom_exclusions:
            excluded["custom"] = sorted({str(n) for n in custom_exclusions})
        return {reason: numbers for reason, numbers in excluded.items() if numbers}

    def _enriched_examiner_numbers(self, patent: Patent) -> set[str]:
        """Examiner-cited art recovered from the file wrapper, best-effort."""
        if self.file_wrapper is None:
            return set()
        try:
            enriched = self.file_wrapper.enrich_patent(patent)
            if isinstance(enriched, Patent):
                return set(enriched.examiner_cited_numbers)
        except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
            logger.warning("file-wrapper enrichment failed for %s: %s", patent.patent_number, exc)
        return set()

    # --------------------------------------------------------------- search
    def search(
        self,
        patent: Patent,
        claims: Optional[list[int]] = None,
        *,
        final_k: int = 25,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Optional[Callable[[TraceStep], None]] = None,
        exclude_examiner_art: bool = True,
        exclude_family: bool = True,
        custom_exclusions: Sequence[str] = (),
        feedback_messages: Sequence[str] = (),
        resume_messages: Optional[list[dict]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> InvaliditySearchResult:
        """Run the agentic prior-art search for ``patent``.

        Args:
            patent: the target patent.
            claims: claim numbers to invalidate (default: independent
                claims, or claim 1).
            final_k: ranked references returned (best first).
            budget_seconds: wall-clock budget for the agent (default ≤3 min).
            max_steps: maximum agent rounds.
            on_step: live callback receiving every
                :class:`~patentkit.llm.tools.TraceStep`.
            exclude_examiner_art: drop examiner-cited references (default on).
            exclude_family: drop same-family publications.
            custom_exclusions: extra patent numbers to drop.
            feedback_messages: user feedback injected as user messages
                (typically together with ``resume_messages``).
            resume_messages: a previous result's conversation (from
                ``trace``/guided session) to resume the SAME agent run.
            progress: optional callback receiving human-readable updates.
        """
        t0 = time.monotonic()
        claims = claims or [c.number for c in patent.independent_claims] or [1]
        cutoff = patent.best_effective_date()
        excluded = self._build_exclusions(patent, exclude_examiner_art, exclude_family,
                                          custom_exclusions)

        if self.llm is None:
            return self._degraded_search(patent, claims, excluded, final_k,
                                         progress=progress, t0=t0)

        def _step(step: TraceStep) -> None:
            report_progress(progress, step_summary(step))
            if on_step is not None:
                on_step(step)

        runner = AgenticSearchRunner(self.keyword_store, self.vector_store, self.llm,
                                     max_steps=max_steps, budget_seconds=budget_seconds)
        report_progress(progress, f"agentic invalidity search: cutoff {cutoff}, "
                                  f"budget {budget_seconds:g}s / {max_steps} steps")
        outcome: AgenticSearchOutcome = runner.run(
            "invalidity",
            target_patent=patent,
            claims=claims,
            exclusions=excluded,
            cutoff_date=cutoff,
            final_k=final_k,
            feedback_messages=feedback_messages,
            resume_messages=resume_messages,
            on_step=_step,
        )
        return InvaliditySearchResult(
            target=str(patent.patent_number),
            claims=claims,
            plan_or_params={
                "mode": "agentic",
                "before_date": cutoff.isoformat() if cutoff else None,
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
    def _degraded_search(self, patent: Patent, claims: list[int],
                         excluded: dict[str, list[str]], final_k: int, *,
                         progress: Optional[Callable[[str], None]],
                         t0: float) -> InvaliditySearchResult:
        """Keys-free fallback: ONE plain BM25 pass with the default exclusions."""
        cutoff = patent.best_effective_date()
        exclude_numbers: list[PatentNumber] = []
        for numbers in excluded.values():
            for raw in numbers:
                try:
                    exclude_numbers.append(PatentNumber.parse(raw))
                except ValueError:
                    logger.warning("Skipping unparseable exclusion number: %r", raw)
        keywords = _fallback_keywords(patent, claims=claims)
        query = SearchQuery(keywords=keywords, before_date=cutoff,
                            exclude_numbers=exclude_numbers, limit=max(final_k, 25))
        report_progress(progress, "degraded keys-free mode: single keyword pass "
                                  f"({len(keywords)} keywords, cutoff {cutoff})")
        hits = self.keyword_store.search(query)
        max_score = max((r.score for r in hits), default=0.0) or 1.0
        results = [result_to_dict(r, r.score / max_score,
                                  "degraded mode: BM25 keyword relevance only")
                   for r in hits[:final_k]]
        return InvaliditySearchResult(
            target=str(patent.patent_number),
            claims=claims,
            plan_or_params={
                "mode": "degraded_keyword_only",
                "before_date": cutoff.isoformat() if cutoff else None,
                "keywords": keywords,
                "final_k": final_k,
            },
            results=results,
            excluded=excluded,
            timing={"total": round(time.monotonic() - t0, 4)},
            trace=None,
            stop_reason="degraded",
        )


__all__ = ["InvaliditySearchAgent", "InvaliditySearchResult",
           "candidates_to_result_dicts", "step_summary"]
