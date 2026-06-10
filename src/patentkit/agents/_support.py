"""Private helpers shared by the search agents.

Everything here is dependency-free (stdlib + the patentkit core) so agents
keep working when the optional analysis / hybrid-search modules are absent.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from patentkit.search.base import Passage, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


def _rrf(result_lists: Sequence[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Local reciprocal-rank fusion: score = sum over lists of 1/(k + rank).

    Mirrors ``patentkit.search.hybrid.rrf_fuse`` so agents do not depend on
    that module being installed/finished. Passages from every contributing
    list are merged (deduplicated by text).
    """
    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            key = str(result.patent_number)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in fused:
                fused[key] = SearchResult(
                    patent_number=result.patent_number,
                    score=0.0,
                    patent=result.patent,
                    passages=list(result.passages),
                    explanation="rrf fusion",
                )
            else:
                kept = fused[key]
                if kept.patent is None and result.patent is not None:
                    kept.patent = result.patent
                seen = {p.text for p in kept.passages}
                kept.passages += [p for p in result.passages if p.text not in seen]
    out = list(fused.values())
    for result in out:
        result.score = scores[str(result.patent_number)]
    out.sort(key=lambda r: r.score, reverse=True)
    return out


def fuse_rankings(result_lists: Sequence[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Fuse rankings with ``patentkit.search.hybrid.rrf_fuse`` when available,
    falling back to the local :func:`_rrf` copy."""
    try:
        from patentkit.search.hybrid import rrf_fuse  # noqa: PLC0415 — lazy by design
        return rrf_fuse(list(result_lists), k=k)
    except Exception as exc:  # noqa: BLE001 — ImportError or an unfinished module
        if not isinstance(exc, ImportError):
            logger.warning("hybrid.rrf_fuse failed (%s); using local RRF", exc)
        return _rrf(result_lists, k=k)


def merge_query(base: SearchQuery, extra: Optional[SearchQuery]) -> SearchQuery:
    """Merge user-supplied ``extra`` query params into agent-built ``base``.

    List fields are unioned (order-preserving); scalar fields from ``extra``
    win when set; ``before_date`` takes the *earlier* of the two so a user
    can only tighten (never loosen) a prior-art cutoff.
    """
    if extra is None:
        return base

    def union(a: list, b: list) -> list:
        seen, out = set(), []
        for item in list(a) + list(b):
            key = str(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    merged = SearchQuery(
        keywords=union(base.keywords, extra.keywords),
        required_keywords=union(base.required_keywords, extra.required_keywords),
        excluded_keywords=union(base.excluded_keywords, extra.excluded_keywords),
        text=extra.text or base.text,
        minimum_match=extra.minimum_match if extra.minimum_match is not None else base.minimum_match,
        fields=extra.fields if extra.fields else base.fields,
        art_classes=union(base.art_classes, extra.art_classes),
        inventors=union(base.inventors, extra.inventors),
        assignees=union(base.assignees, extra.assignees),
        countries=union(base.countries, extra.countries),
        include_numbers=union(base.include_numbers, extra.include_numbers),
        exclude_numbers=union(base.exclude_numbers, extra.exclude_numbers),
        limit=max(base.limit, extra.limit),
        after_date=extra.after_date or base.after_date,
    )
    dates = [d for d in (base.before_date, extra.before_date) if d]
    merged.before_date = min(dates) if dates else None
    return merged


def llm_relevance_scores(
    llm,
    candidates: list[SearchResult],
    target_text: str,
    *,
    task: str = "anticipates or renders obvious the claims",
) -> dict[str, tuple[float, str]]:
    """Single batched LLM call scoring each candidate 0-10 against a target.

    Returns ``{patent_number: (score_0_to_10, why)}``; an empty dict on any
    LLM/JSON failure (callers then keep the pre-LLM ranking). One call over
    title + best passages keeps token cost low versus per-candidate calls.
    """
    if llm is None or not candidates:
        return {}
    lines = []
    for result in candidates:
        passages = " | ".join(p.text[:280].replace("\n", " ") for p in result.passages[:2])
        title = result.patent.title if result.patent else None
        lines.append(f'- number: "{result.patent_number}" title: "{title or "?"}" passages: "{passages}"')
    prompt = (
        f"Target text:\n{target_text[:4000]}\n\n"
        f"Score each candidate document 0-10 on how strongly it {task} above. "
        "Respond with ONLY a JSON array of objects "
        '[{"number": "...", "score": <0-10>, "why": "<one sentence>"}].\n\n'
        "Candidates:\n" + "\n".join(lines)
    )
    try:
        raw = llm.complete_json(prompt, max_tokens=4096)
        if isinstance(raw, dict):
            raw = raw.get("scores", raw.get("results", []))
        out: dict[str, tuple[float, str]] = {}
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or "number" not in item:
                continue
            try:
                score = max(0.0, min(10.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                continue
            out[str(item["number"])] = (score, str(item.get("why", "")))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batched LLM relevance scoring failed (%s); keeping pre-LLM ranking", exc)
        return {}


def combine_scores(
    candidates: list[SearchResult],
    llm_scores: dict[str, tuple[float, str]],
    *,
    llm_weight: float = 0.75,
    stage_weight: float = 0.25,
) -> list[tuple[SearchResult, float, Optional[str]]]:
    """Combine 0.75 x (LLM score / 10) with 0.25 x max-normalized stage score.

    Mirrors the production zscore_combined weighting (0.75 disclosure +
    0.25 keyword). Without LLM scores the normalized stage score alone is
    used, preserving the input ordering.
    """
    max_stage = max((r.score for r in candidates), default=0.0) or 1.0
    out: list[tuple[SearchResult, float, Optional[str]]] = []
    for result in candidates:
        norm = result.score / max_stage
        key = str(result.patent_number)
        if llm_scores:
            llm_score, why = llm_scores.get(key, (0.0, None))
            combined = llm_weight * (llm_score / 10.0) + stage_weight * norm
        else:
            combined, why = norm, None
        out.append((result, combined, why))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def result_to_dict(result: SearchResult, score: float, why: Optional[str]) -> dict:
    """Serialize one ranked result for the pydantic result models."""
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


def report_progress(progress: Optional[Callable[[str], None]], message: str) -> None:
    """Invoke a user progress callback, never letting it break the pipeline."""
    logger.info(message)
    if progress is not None:
        try:
            progress(message)
        except Exception:  # noqa: BLE001 — user callback must not kill a search
            logger.exception("progress callback raised")
