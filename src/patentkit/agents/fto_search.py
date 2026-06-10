"""Freedom-to-operate (FTO) search agent.

Given a product description, finds potentially in-force patents whose claims
the product might practice. Keywords come from a MEDIUM-effort LLM when
available, otherwise from a token-frequency heuristic, so the agent works
keys-free.

**In-force approximation**: when ``in_force_only`` is set the query is
restricted to ``after_date = today - 21 years``. US utility patents expire
20 years from the earliest non-provisional filing date; the extra year
absorbs provisional priority and patent-term adjustment. This is a recall
filter only — actual term/fee status must be verified per patent.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Callable, Optional

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
    #: ranked patents: {patent_number, title, score, passages, why}
    results: list[dict] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    requires_attorney_review: bool = True


class FtoSearchAgent:
    """Keyword (+ optional vector / LLM) FTO screening over a patent store."""

    def __init__(self, keyword_store: KeywordStore, vector_store: Optional[VectorStore] = None, llm=None):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.llm = llm

    def _keywords(self, product_description: str) -> list[str]:
        """Keywords from a MEDIUM-effort LLM, else the local token heuristic."""
        if self.llm is not None:
            prompt = (
                "Extract 8-15 short technical search keywords/phrases that would "
                "retrieve patents covering this product. Respond with ONLY a JSON "
                f"array of strings.\n\nProduct: {product_description[:3000]}"
            )
            try:
                raw = self.llm.complete_json(prompt, max_tokens=1024)
                keywords = [str(k) for k in raw if isinstance(k, (str, int))] if isinstance(raw, list) else []
                if keywords:
                    return keywords[:15]
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM keyword generation failed (%s); using fallback", exc)
        return _fallback_keywords(product_description=product_description)

    def search(
        self,
        product_description: str,
        *,
        jurisdiction: str = "US",
        in_force_only: bool = True,
        extra_query: Optional[SearchQuery] = None,
        final_k: int = 25,
        progress: Optional[Callable[[str], None]] = None,
    ) -> FtoSearchResult:
        """Screen the corpus for patents the product might practice.

        Args:
            product_description: free-text description of the product/feature.
            jurisdiction: country code filter (default "US").
            in_force_only: restrict to patents filed within the last
                :data:`IN_FORCE_YEARS` years (documented approximation above).
            extra_query: user query merged into the generated query.
            final_k: results returned.
            progress: optional callback receiving stage updates.
        """
        t0 = time.monotonic()
        keywords = self._keywords(product_description)
        after = date.today() - timedelta(days=int(IN_FORCE_YEARS * 365.25)) if in_force_only else None
        base_query = SearchQuery(
            keywords=keywords,
            text=product_description,
            after_date=after,
            countries=[jurisdiction] if jurisdiction else [],
            limit=max(final_k * 4, 100),
        )
        query = merge_query(base_query, extra_query)
        report_progress(progress, f"FTO: keyword search ({len(query.keywords)} keywords, "
                                  f"after {query.after_date})")
        keyword_results = self.keyword_store.search(query)

        if self.vector_store is not None and keyword_results:
            vector_results = self.vector_store.search_text(product_description,
                                                           limit=query.limit, query=query)
            candidates = fuse_rankings([keyword_results, vector_results])[: query.limit]
        else:
            candidates = keyword_results

        llm_scores = llm_relevance_scores(
            self.llm, candidates[: max(final_k * 2, 50)], product_description,
            task="claims subject matter this product likely practices",
        )
        ranked = combine_scores(candidates, llm_scores)

        return FtoSearchResult(
            product_description=product_description,
            jurisdiction=jurisdiction,
            in_force_only=in_force_only,
            plan_or_params={
                "keywords": query.keywords,
                "after_date": query.after_date.isoformat() if query.after_date else None,
                "countries": query.countries,
                "final_k": final_k,
                "llm_rerank": bool(llm_scores),
            },
            results=[result_to_dict(r, score, why) for r, score, why in ranked[:final_k]],
            timing={"total": round(time.monotonic() - t0, 4)},
        )


__all__ = ["FtoSearchAgent", "FtoSearchResult", "IN_FORCE_YEARS"]
