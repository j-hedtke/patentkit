"""Hybrid search: fuse keyword and vector rankings.

:func:`rrf_fuse` implements reciprocal rank fusion (RRF) over any number of
ranked :class:`~patentkit.search.base.SearchResult` lists, merging passages
and keeping the richest patent object per document. :class:`HybridSearcher`
runs a keyword store and a vector store side by side and fuses the results.
"""

from __future__ import annotations

import logging
import statistics

from patentkit.models import PatentNumber
from patentkit.search.base import KeywordStore, SearchQuery, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _doc_key(number: PatentNumber) -> str:
    """Kind-code-insensitive identity so B1/B2 variants fuse together."""
    return f"{number.country_code}{number.number.lstrip('0') or '0'}"


def rrf_fuse(result_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Reciprocal rank fusion of multiple ranked result lists.

    Each appearance of a document at (0-based) rank ``r`` contributes
    ``1 / (k + r + 1)`` to its fused score. Documents are identified
    kind-code-insensitively; passages from all appearances are merged
    (deduplicated by field+text) and the best (first non-None, preferring
    richer) patent object is kept.

    Args:
        result_lists: ranked result lists (e.g. [keyword results, vector results]).
        k: the standard RRF damping constant (default 60).

    Returns:
        Fused results sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    fused: dict[str, SearchResult] = {}
    sources: dict[str, list[str]] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            key = _doc_key(result.patent_number)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in fused:
                fused[key] = SearchResult(
                    patent_number=result.patent_number,
                    score=0.0,
                    patent=result.patent,
                    passages=list(result.passages),
                    explanation=None,
                )
                sources[key] = []
            else:
                merged = fused[key]
                if merged.patent is None and result.patent is not None:
                    merged.patent = result.patent
                seen = {(p.field, p.text) for p in merged.passages}
                for passage in result.passages:
                    if (passage.field, passage.text) not in seen:
                        merged.passages.append(passage)
                        seen.add((passage.field, passage.text))
            if result.explanation:
                sources[key].append(result.explanation)

    for key, result in fused.items():
        result.score = scores[key]
        joined = "; ".join(dict.fromkeys(sources[key])) or "rrf"
        result.explanation = f"rrf(k={k}): {joined}"

    return sorted(fused.values(), key=lambda r: r.score, reverse=True)


def zscore_combine(scored: dict[str, list[float]],
                   weights: dict[str, float] | None = None) -> list[float]:
    """Combine aligned score lists from multiple scoring systems by z-score.

    Each system's scores are standardized to zero mean / unit variance and
    summed (optionally weighted), making heterogeneous score scales (BM25 vs
    cosine similarity) comparable.

    Args:
        scored: system name -> scores, aligned by document position; all
            lists must have the same length.
        weights: optional system name -> weight (default 1.0 each).

    Returns:
        One combined score per document position.
    """
    lengths = {len(values) for values in scored.values()}
    if len(lengths) > 1:
        raise ValueError(f"Score lists must be aligned; got lengths {sorted(lengths)}")
    n = lengths.pop() if lengths else 0
    combined = [0.0] * n
    for system, values in scored.items():
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        weight = (weights or {}).get(system, 1.0)
        for i, value in enumerate(values):
            z = (value - mean) / std if std > 0 else 0.0
            combined[i] += weight * z
    return combined


class HybridSearcher:
    """Run keyword + vector search over the same query and RRF-fuse the results.

    The vector leg embeds ``query.text`` when present, otherwise the joined
    ``query.keywords``; metadata filters (dates, classes, exclusions) are
    passed through to both stores.
    """

    def __init__(self, keyword_store: KeywordStore, vector_store: VectorStore, k: int = 60):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.k = k

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Fused results, truncated to ``query.limit``."""
        keyword_results = self.keyword_store.search(query)
        text = query.text or " ".join(query.keywords)
        if text.strip():
            vector_results = self.vector_store.search_text(
                text, limit=query.limit, query=query
            )
        else:
            logger.debug("Hybrid search skipped vector leg: no text or keywords")
            vector_results = []
        fused = rrf_fuse([keyword_results, vector_results], k=self.k)
        return fused[: query.limit]
