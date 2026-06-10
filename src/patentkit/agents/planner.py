"""Agentic search planning.

``plan_search`` asks a HIGH-effort LLM to decompose an invalidity / FTO /
infringement task into several complementary keyword queries (a
:class:`SearchPlan`), with a documented run-time estimate attached. Every
path degrades gracefully when no LLM (or no API key) is available: the plan
falls back to a single keyword query built from the most distinctive
title/claim terms (``_fallback_keywords``), so the toolkit works keys-free.

:class:`QuerySpec` is a pydantic mirror of the
:class:`~patentkit.search.base.SearchQuery` dataclass so plans serialize
cleanly through MCP / OpenAI tool calls.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from patentkit.agents.feedback import SearchFeedback
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import SearchQuery

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: claim/title boilerplate that never makes a good search keyword
_STOPWORDS = frozenset("""
a an and are as at be being by claim claims comprise comprises comprising
configured each first for from further has have having herein in includes
including is it its least method methods of on one or other plurality
provided said second such system systems that the thereof third to use used
using via wherein whereby which with
""".split())

#: corpus size assumed when none is known (used for the initial plan estimate)
DEFAULT_CORPUS_SIZE = 100_000
#: number of candidates assumed to flow into an LLM rerank stage
DEFAULT_RERANK_CANDIDATES = 100


class QuerySpec(BaseModel):
    """Pydantic-serializable mirror of the :class:`SearchQuery` dataclass.

    ``SearchQuery`` is a plain dataclass with ``PatentNumber`` fields, so
    plans store this mirror instead and convert at execution time via
    :meth:`to_search_query` / :meth:`from_search_query`. Patent numbers are
    held as plain strings.
    """

    keywords: list[str] = Field(default_factory=list)
    required_keywords: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)
    text: Optional[str] = None
    minimum_match: Optional[int] = None
    fields: list[str] = Field(default_factory=lambda: ["title", "abstract", "claims", "specification"])
    art_classes: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    before_date: Optional[date] = None
    after_date: Optional[date] = None
    countries: list[str] = Field(default_factory=list)
    include_numbers: list[str] = Field(default_factory=list)
    exclude_numbers: list[str] = Field(default_factory=list)
    limit: int = 100

    def to_search_query(self) -> SearchQuery:
        """Convert to the dataclass consumed by search stores.

        Unparseable patent numbers are skipped with a log warning rather
        than failing the whole query.
        """
        return SearchQuery(
            keywords=list(self.keywords),
            required_keywords=list(self.required_keywords),
            excluded_keywords=list(self.excluded_keywords),
            text=self.text,
            minimum_match=self.minimum_match,
            fields=list(self.fields),
            art_classes=list(self.art_classes),
            inventors=list(self.inventors),
            assignees=list(self.assignees),
            before_date=self.before_date,
            after_date=self.after_date,
            countries=list(self.countries),
            include_numbers=_parse_numbers(self.include_numbers),
            exclude_numbers=_parse_numbers(self.exclude_numbers),
            limit=self.limit,
        )

    @classmethod
    def from_search_query(cls, query: SearchQuery) -> "QuerySpec":
        """Build a spec from an existing :class:`SearchQuery`."""
        return cls(
            keywords=list(query.keywords),
            required_keywords=list(query.required_keywords),
            excluded_keywords=list(query.excluded_keywords),
            text=query.text,
            minimum_match=query.minimum_match,
            fields=list(query.fields),
            art_classes=list(query.art_classes),
            inventors=list(query.inventors),
            assignees=list(query.assignees),
            before_date=query.before_date,
            after_date=query.after_date,
            countries=list(query.countries),
            include_numbers=[str(n) for n in query.include_numbers],
            exclude_numbers=[str(n) for n in query.exclude_numbers],
            limit=query.limit,
        )


def _parse_numbers(raw: list[str]) -> list[PatentNumber]:
    out: list[PatentNumber] = []
    for item in raw:
        try:
            out.append(PatentNumber.parse(item))
        except ValueError:
            logger.warning("Skipping unparseable patent number in query spec: %r", item)
    return out


class PlannedQuery(BaseModel):
    """One query of a search plan, with the reason it was included."""

    purpose: str
    query: QuerySpec


class SearchPlan(BaseModel):
    """An agent-produced plan: complementary queries plus exclusions."""

    search_type: Literal["invalidity", "fto", "infringement"]
    target: str
    rationale: str
    queries: list[PlannedQuery] = Field(default_factory=list)
    #: patent numbers to exclude from results (e.g. examiner-cited art)
    exclusions: list[str] = Field(default_factory=list)
    estimated_seconds: Optional[float] = None


def _fallback_keywords(
    patent: Patent | None = None,
    product_description: str | None = None,
    claims: list[int] | None = None,
    max_keywords: int = 8,
) -> list[str]:
    """Keys-free keyword extraction: most frequent distinctive terms.

    Pulls tokens from the patent title and (selected) claim texts — or from a
    product description — drops claim boilerplate stopwords and short tokens,
    and returns the most frequent remainder, title terms first.
    """
    title_tokens: list[str] = []
    body_tokens: list[str] = []
    if patent is not None:
        title_tokens = _TOKEN_RE.findall((patent.title or "").lower())
        selected = patent.claims
        if claims:
            selected = [c for c in patent.claims if c.number in set(claims)] or patent.claims
        for claim in selected:
            body_tokens += _TOKEN_RE.findall(claim.text.lower())
        body_tokens += _TOKEN_RE.findall((patent.abstract or "").lower())
    if product_description:
        body_tokens += _TOKEN_RE.findall(product_description.lower())

    def keep(tok: str) -> bool:
        return len(tok) > 2 and tok not in _STOPWORDS and not tok.isdigit()

    counts = Counter(t for t in body_tokens if keep(t))
    ordered: list[str] = []
    for tok in title_tokens:  # title terms are the most distinctive — lead with them
        if keep(tok) and tok not in ordered:
            ordered.append(tok)
    for tok, _ in counts.most_common():
        if tok not in ordered:
            ordered.append(tok)
    return ordered[:max_keywords]


_PLAN_PROMPT = """You are planning a {search_type} prior-art/patent search.
Target:
{target_block}

Produce a JSON array of {n_queries} complementary search query specs. Each item:
{{"purpose": "<why this query>", "keywords": [...], "required_keywords": [...],
  "excluded_keywords": [...], "art_classes": ["CPC prefixes"],
  "before_date": "YYYY-MM-DD" or null, "after_date": "YYYY-MM-DD" or null,
  "text": "optional free-text semantic query"}}
Vary the angle across queries (synonyms, adjacent technology, implementation
details, problem-oriented phrasing). Respond with ONLY the JSON array."""


def _target_block(patent: Patent | None, product_description: str | None,
                  claims: list[int] | None) -> str:
    parts: list[str] = []
    if patent is not None:
        parts.append(f"Patent {patent.patent_number}: {patent.title or '(untitled)'}")
        if patent.abstract:
            parts.append(f"Abstract: {patent.abstract[:600]}")
        if patent.cpc_codes:
            parts.append("CPC: " + ", ".join(patent.cpc_codes[:8]))
        selected = patent.claims
        if claims:
            selected = [c for c in patent.claims if c.number in set(claims)] or patent.claims
        for claim in selected[:4]:
            parts.append(f"Claim {claim.number}: {claim.text[:800]}")
    if product_description:
        parts.append(f"Product description: {product_description[:1200]}")
    return "\n".join(parts) or "(no target details provided)"


def _coerce_query_spec(item: dict, *, before_date: date | None) -> QuerySpec | None:
    """Validate/coerce one LLM-emitted query dict into a QuerySpec."""
    if not isinstance(item, dict):
        return None
    allowed = set(QuerySpec.model_fields)
    cleaned = {k: v for k, v in item.items() if k in allowed and v is not None}
    for key in ("keywords", "required_keywords", "excluded_keywords", "art_classes"):
        if key in cleaned and isinstance(cleaned[key], str):
            cleaned[key] = [cleaned[key]]
    try:
        spec = QuerySpec(**cleaned)
    except Exception:  # pydantic ValidationError without importing it here
        logger.warning("Discarding malformed query spec from LLM: %r", item)
        return None
    if before_date and spec.before_date is None:
        spec.before_date = before_date
    if not spec.keywords and not spec.text and not spec.required_keywords:
        return None
    return spec


def plan_search(
    search_type: Literal["invalidity", "fto", "infringement"],
    *,
    patent: Patent | None = None,
    product_description: str | None = None,
    claims: list[int] | None = None,
    llm=None,
    n_queries: int = 4,
) -> SearchPlan:
    """Plan a multi-query search with a HIGH-effort LLM.

    The LLM is prompted (seeded with the patent title/claims/CPC codes or the
    product description) for a JSON array of query specs. On any LLM or JSON
    failure — including ``llm=None`` — the plan falls back to a single
    keyword query built by :func:`_fallback_keywords`, so planning always
    succeeds without API keys.

    Args:
        search_type: "invalidity", "fto", or "infringement".
        patent: the target patent (invalidity / infringement).
        product_description: the target product (FTO).
        claims: claim numbers in focus (default: all/independent).
        llm: an :class:`patentkit.llm.LLM`; pass ``get_llm("high")`` in
            production. ``None`` skips the LLM entirely.
        n_queries: how many complementary queries to request.

    Returns:
        A :class:`SearchPlan` with ``estimated_seconds`` attached (computed
        against :data:`DEFAULT_CORPUS_SIZE`; recompute with
        :func:`estimate_search_seconds` once the real corpus size is known).
    """
    target = str(patent.patent_number) if patent else (product_description or "")[:80]
    before = patent.best_effective_date() if (patent and search_type == "invalidity") else None
    exclusions = sorted(patent.examiner_cited_numbers) if (patent and search_type == "invalidity") else []

    queries: list[PlannedQuery] = []
    rationale = ""
    if llm is not None:
        prompt = _PLAN_PROMPT.format(
            search_type=search_type,
            target_block=_target_block(patent, product_description, claims),
            n_queries=n_queries,
        )
        try:
            raw = llm.complete_json(prompt, max_tokens=2048)
            if isinstance(raw, dict):  # tolerate {"queries": [...]}
                raw = raw.get("queries", [raw])
            for item in raw if isinstance(raw, list) else []:
                spec = _coerce_query_spec(item, before_date=before)
                if spec is not None:
                    purpose = str(item.get("purpose", "planned query")) if isinstance(item, dict) else "planned query"
                    queries.append(PlannedQuery(purpose=purpose, query=spec))
            rationale = f"LLM-planned {len(queries)} complementary queries."
        except Exception as exc:  # noqa: BLE001 — any LLM/JSON failure falls back
            logger.warning("plan_search LLM planning failed (%s); using fallback keywords", exc)
            queries = []

    if not queries:
        keywords = _fallback_keywords(patent, product_description, claims)
        queries = [PlannedQuery(
            purpose="fallback keyword query from distinctive title/claim terms",
            query=QuerySpec(keywords=keywords, before_date=before),
        )]
        rationale = "Heuristic fallback plan (no LLM available or LLM output unusable)."

    plan = SearchPlan(
        search_type=search_type,
        target=target,
        rationale=rationale,
        queries=queries,
        exclusions=exclusions,
    )
    plan.estimated_seconds = estimate_search_seconds(
        plan, corpus_size=DEFAULT_CORPUS_SIZE, with_llm_rerank=llm is not None,
    )
    return plan


def estimate_search_seconds(
    plan: SearchPlan,
    corpus_size: int,
    with_llm_rerank: bool,
    charting_claims: int = 0,
) -> float:
    """Estimate wall-clock seconds for executing ``plan``.

    Heuristic (all constants are rough single-machine figures):

    - **per-query base**: 2.0 s setup/IO per planned query;
    - **corpus factor**: 0.6 s x log10(corpus_size + 10) per query — index
      scans grow roughly logarithmically with corpus size for both BM25
      heaps and ANN search;
    - **LLM rerank**: a single batched relevance call costs ~8 s overhead
      plus ~0.12 s per candidate (prompt-size growth), candidates capped at
      :data:`DEFAULT_RERANK_CANDIDATES`;
    - **charting**: ~45 s per claim charted (one HIGH-effort LLM pass per
      claim across references).

    The estimate is monotonically non-decreasing in every argument.
    """
    n_queries = max(1, len(plan.queries))
    seconds = n_queries * (2.0 + 0.6 * math.log10(corpus_size + 10))
    if with_llm_rerank:
        seconds += 8.0 + 0.12 * min(corpus_size, DEFAULT_RERANK_CANDIDATES)
    seconds += 45.0 * max(0, charting_claims)
    return round(seconds, 1)


def humanize_seconds(s: float) -> str:
    """Render a duration for humans: '45 seconds', '3.5 minutes', '1.2 hours'."""
    if s < 90:
        return f"{int(round(s))} seconds"
    if s < 5400:
        return f"{s / 60:.1f} minutes"
    return f"{s / 3600:.1f} hours"


_REVISE_PROMPT = """You previously produced this {search_type} search plan (JSON):
{plan_json}

The user gave this feedback:
{feedback}

Produce a REVISED JSON array of query specs (same schema: purpose, keywords,
required_keywords, excluded_keywords, art_classes, before_date, after_date,
text). Keep queries the user liked, fix or drop the ones criticized, and
incorporate the free-text guidance. Respond with ONLY the JSON array."""


def revise_plan(plan: SearchPlan, feedback: SearchFeedback, llm=None) -> SearchPlan:
    """Revise a plan from user feedback (HIGH-effort LLM when available).

    Without an LLM (or on LLM failure) deterministic heuristics apply:
    queries judged ``off_topic`` are dropped, ``too_broad`` queries get a
    tighter ``minimum_match``, ``too_narrow`` ones get it relaxed, and
    patents marked irrelevant are added to the plan's exclusions.
    """
    revised = plan.model_copy(deep=True)
    if llm is not None:
        try:
            prompt = _REVISE_PROMPT.format(
                search_type=plan.search_type,
                plan_json=json.dumps(plan.model_dump(mode="json"), indent=1)[:6000],
                feedback=feedback.summary_for_llm(),
            )
            raw = llm.complete_json(prompt, max_tokens=2048)
            if isinstance(raw, dict):
                raw = raw.get("queries", [raw])
            before = next((q.query.before_date for q in plan.queries if q.query.before_date), None)
            queries: list[PlannedQuery] = []
            for item in raw if isinstance(raw, list) else []:
                spec = _coerce_query_spec(item, before_date=before)
                if spec is not None:
                    purpose = str(item.get("purpose", "revised query")) if isinstance(item, dict) else "revised query"
                    queries.append(PlannedQuery(purpose=purpose, query=spec))
            if queries:
                revised.queries = queries
                revised.rationale = (plan.rationale + " | revised from user feedback").strip(" |")
                _apply_exclusion_feedback(revised, feedback)
                return revised
        except Exception as exc:  # noqa: BLE001
            logger.warning("revise_plan LLM revision failed (%s); applying heuristics", exc)

    # Heuristic revision path.
    verdicts = {q.query_index: q.verdict for q in feedback.queries}
    kept: list[PlannedQuery] = []
    for i, planned in enumerate(revised.queries):
        verdict = verdicts.get(i)
        if verdict == "off_topic":
            continue
        if verdict == "too_broad":
            spec = planned.query
            spec.minimum_match = min(len(spec.keywords) or 1, (spec.minimum_match or spec.to_search_query().effective_minimum_match()) + 1)
        elif verdict == "too_narrow":
            spec = planned.query
            spec.minimum_match = max(1, (spec.minimum_match or 2) - 1)
        kept.append(planned)
    if kept:
        revised.queries = kept
    _apply_exclusion_feedback(revised, feedback)
    revised.rationale = (revised.rationale + " | heuristically revised from feedback").strip(" |")
    return revised


def _apply_exclusion_feedback(plan: SearchPlan, feedback: SearchFeedback) -> None:
    """Add results the user marked irrelevant to the plan's exclusions."""
    for r in feedback.results:
        if r.relevant is False and r.patent_number not in plan.exclusions:
            plan.exclusions.append(r.patent_number)


__all__ = [
    "QuerySpec",
    "PlannedQuery",
    "SearchPlan",
    "plan_search",
    "revise_plan",
    "estimate_search_seconds",
    "humanize_seconds",
    "_fallback_keywords",
]
