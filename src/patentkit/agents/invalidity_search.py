"""Three-stage invalidity (prior-art) search agent.

A clean reimplementation of the production pipeline:

- **stage 1** — broad keyword search (k=1000) with the prior-art date cutoff
  (``before_date = patent.best_effective_date()``) and examiner-art / family /
  self exclusions applied up front;
- **stage 2** — semantic rerank: when a vector store is configured the
  keyword and vector rankings are fused with reciprocal-rank fusion
  (``patentkit.search.hybrid.rrf_fuse`` when installed, else a local copy);
- **stage 3** — LLM relevance scoring (only when an LLM is configured): one
  batched JSON call scores each candidate 0-10 against the target claims,
  combined as ``0.75 * llm + 0.25 * normalized stage-2 score`` (mirroring the
  production 0.75 disclosure / 0.25 keyword weighting).

Every stage degrades gracefully: with no vector store stage 2 keeps the
keyword ranking; with no LLM stage 3 is skipped — the agent always works
keys-free.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.agents._support import (
    combine_scores,
    fuse_rankings,
    llm_relevance_scores,
    merge_query,
    report_progress,
    result_to_dict,
)
from patentkit.agents.planner import _fallback_keywords
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery, SearchResult, VectorStore

logger = logging.getLogger(__name__)

#: stage-1 breadth, matching the production pipeline's k=1000
STAGE1_K = 1000


class InvaliditySearchResult(BaseModel):
    """Serializable outcome of one invalidity search run."""

    target: str
    claims: list[int] = Field(default_factory=list)
    #: the query parameters / plan actually executed (for reproducibility)
    plan_or_params: dict = Field(default_factory=dict)
    #: ranked references: {patent_number, title, score, passages, why}
    results: list[dict] = Field(default_factory=list)
    #: exclusion reason -> patent numbers excluded for that reason
    excluded: dict[str, list[str]] = Field(default_factory=dict)
    #: per-stage wall-clock seconds
    timing: dict[str, float] = Field(default_factory=dict)


class InvaliditySearchAgent:
    """Runs the 3-stage invalidity pipeline against pluggable stores.

    Args:
        keyword_store: any :class:`~patentkit.search.base.KeywordStore`.
        vector_store: optional :class:`~patentkit.search.base.VectorStore`
            enabling the stage-2 semantic rerank.
        llm: optional :class:`patentkit.llm.LLM` enabling LLM keyword
            generation and stage-3 relevance scoring.
        file_wrapper: optional client with an ``enrich_patent(patent)``
            method (e.g. ``patentkit.connectors.inference.file_wrapper.
            FileWrapperClient``) used to recover examiner-cited art missing
            from the patent record; failures are logged and skipped.
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
        """Default exclusion sets, keyed by reason (mirrors should_exclude())."""
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

    # ------------------------------------------------------------- keywords
    def _keywords(self, patent: Patent, claims: list[int], claims_text: str) -> list[str]:
        """Search keywords from the LLM (LOW-effort task) or local fallback."""
        if self.llm is not None:
            prompt = (
                "Extract 8-15 short technical search keywords/phrases that "
                "would retrieve prior art for these patent claims. Respond "
                "with ONLY a JSON array of strings.\n\n"
                f"Title: {patent.title or ''}\nClaims:\n{claims_text[:5000]}"
            )
            try:
                raw = self.llm.complete_json(prompt, max_tokens=1024)
                keywords = [str(k) for k in raw if isinstance(k, (str, int))] if isinstance(raw, list) else []
                if keywords:
                    return keywords[:15]
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM keyword generation failed (%s); using fallback", exc)
        return _fallback_keywords(patent, claims=claims)

    # --------------------------------------------------------------- search
    def search(
        self,
        patent: Patent,
        claims: Optional[list[int]] = None,
        *,
        extra_query: Optional[SearchQuery] = None,
        exclude_examiner_art: bool = True,
        exclude_family: bool = True,
        custom_exclusions: Sequence[str] = (),
        stage2_k: int = 100,
        final_k: int = 25,
        progress: Optional[Callable[[str], None]] = None,
    ) -> InvaliditySearchResult:
        """Run the full pipeline for ``patent`` and return ranked prior art.

        Args:
            patent: the target patent.
            claims: claim numbers to invalidate (default: independent claims,
                or claim 1).
            extra_query: user query merged into the stage-1 query (keywords
                unioned; the prior-art cutoff can only be tightened).
            exclude_examiner_art: drop examiner-cited references (default on;
                pass ``False`` to search over them too).
            exclude_family: drop same-family publications.
            custom_exclusions: extra patent numbers to drop.
            stage2_k: candidates kept into the rerank/scoring stages.
            final_k: results returned.
            progress: optional callback receiving human-readable stage updates.
        """
        t0 = time.monotonic()
        timing: dict[str, float] = {}
        claims = claims or [c.number for c in patent.independent_claims] or [1]
        selected = [c for c in patent.claims if c.number in set(claims)]
        claims_text = "\n".join(c.text for c in selected) or patent.text_for_search()[:4000]

        excluded = self._build_exclusions(patent, exclude_examiner_art, exclude_family, custom_exclusions)
        exclude_numbers: list[PatentNumber] = []
        for numbers in excluded.values():
            for raw in numbers:
                try:
                    exclude_numbers.append(PatentNumber.parse(raw))
                except ValueError:
                    logger.warning("Skipping unparseable exclusion number: %r", raw)

        # -- stage 1: broad keyword search with date cutoff + exclusions
        keywords = self._keywords(patent, claims, claims_text)
        base_query = SearchQuery(
            keywords=keywords,
            before_date=patent.best_effective_date(),
            exclude_numbers=exclude_numbers,
            limit=STAGE1_K,
        )
        query = merge_query(base_query, extra_query)
        report_progress(progress, f"stage 1: keyword search ({len(query.keywords)} keywords, "
                                  f"cutoff {query.before_date})")
        stage1 = self.keyword_store.search(query)
        timing["stage1"] = round(time.monotonic() - t0, 4)
        report_progress(progress, f"stage 1: {len(stage1)} candidates")

        # -- stage 2: semantic rerank via RRF fusion when a vector store exists
        t1 = time.monotonic()
        if self.vector_store is not None and stage1:
            vector_results = self.vector_store.search_text(claims_text, limit=stage2_k, query=query)
            candidates = fuse_rankings([stage1, vector_results])[:stage2_k]
            report_progress(progress, f"stage 2: fused keyword+vector rankings ({len(candidates)} kept)")
        else:
            candidates = stage1[:stage2_k]
        timing["stage2"] = round(time.monotonic() - t1, 4)

        # -- stage 3: batched LLM relevance scoring (skipped without an LLM)
        t2 = time.monotonic()
        llm_scores = llm_relevance_scores(self.llm, candidates, claims_text)
        ranked = combine_scores(candidates, llm_scores)
        timing["stage3"] = round(time.monotonic() - t2, 4)
        if llm_scores:
            report_progress(progress, f"stage 3: LLM scored {len(llm_scores)} candidates")
        else:
            report_progress(progress, "stage 3: skipped (no LLM) — keeping stage-2 ranking")

        timing["total"] = round(time.monotonic() - t0, 4)
        return InvaliditySearchResult(
            target=str(patent.patent_number),
            claims=claims,
            plan_or_params={
                "keywords": query.keywords,
                "required_keywords": query.required_keywords,
                "excluded_keywords": query.excluded_keywords,
                "before_date": query.before_date.isoformat() if query.before_date else None,
                "stage1_k": query.limit,
                "stage2_k": stage2_k,
                "final_k": final_k,
                "llm_rerank": bool(llm_scores),
                "vector_rerank": self.vector_store is not None,
            },
            results=[result_to_dict(r, score, why) for r, score, why in ranked[:final_k]],
            excluded=excluded,
            timing=timing,
        )


__all__ = ["InvaliditySearchAgent", "InvaliditySearchResult", "STAGE1_K"]
