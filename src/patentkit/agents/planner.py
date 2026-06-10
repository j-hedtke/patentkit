"""Search planning: the cheap pre-run preview shown to users.

Since the search agents moved to the pure agentic core (the LLM generates
and refines its own queries at execution time, inside one tool-use
conversation), the :class:`SearchPlan` is no longer "the queries that will
be executed" — it is the *pre-run user-visible contract*: a deterministic
preview of the initial query angles plus a wall-clock estimate.

**Design choice (documented):** :func:`plan_search` derives the preview from
the same inputs the agent's system prompt gets (title/claim terms, CPC
codes, product description) WITHOUT any LLM call, so starting a guided
session is instant and keys-free. The agent is free to (and expected to)
go beyond these angles at execution time.

:class:`QuerySpec` is a pydantic mirror of the
:class:`~patentkit.search.base.SearchQuery` dataclass so plans serialize
cleanly through MCP / OpenAI tool calls.
"""

from __future__ import annotations

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

#: one agent round ≈ model latency + tool execution (rough single-machine figure)
PER_STEP_LLM_SECONDS = 5.0
#: agent rounds expected beyond the per-angle searches (read task, inspect,
#: shortlist, finish)
BASE_AGENT_STEPS = 4


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
    """One preview angle of a search plan, with the reason it was included."""

    purpose: str
    query: QuerySpec


class SearchPlan(BaseModel):
    """The pre-run contract: initial query angles + exclusions + estimate.

    The agentic searcher treats these as starting guidance only; it
    generates, executes, and refines its own queries at run time.
    """

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


def plan_search(
    search_type: Literal["invalidity", "fto", "infringement"],
    *,
    patent: Patent | None = None,
    product_description: str | None = None,
    claims: list[int] | None = None,
    llm=None,
    n_queries: int = 3,
) -> SearchPlan:
    """Derive the pre-run plan preview — deterministically, no LLM call.

    Up to ``n_queries`` complementary starting angles are derived from the
    same inputs the agent's system prompt gets: (1) broad distinctive
    title/claim (or product) terms, (2) a tighter query requiring the most
    distinctive title terms, (3) a CPC-class-constrained query when the
    target carries classifications. The executing agent treats these as
    guidance only and generates its own queries.

    Args:
        search_type: "invalidity", "fto", or "infringement".
        patent: the target patent (invalidity / infringement).
        product_description: the target product (FTO).
        claims: claim numbers in focus (default: all/independent).
        llm: only used to decide whether the run will be agentic (affects
            the time estimate); never called here.
        n_queries: maximum preview angles.

    Returns:
        A :class:`SearchPlan` with ``estimated_seconds`` attached (computed
        against :data:`DEFAULT_CORPUS_SIZE`; recompute with
        :func:`estimate_search_seconds` once the real corpus size is known).
    """
    target = str(patent.patent_number) if patent else (product_description or "")[:80]
    before = patent.best_effective_date() if (patent and search_type == "invalidity") else None
    exclusions = sorted(patent.examiner_cited_numbers) if (patent and search_type == "invalidity") else []

    keywords = _fallback_keywords(patent, product_description, claims)
    queries: list[PlannedQuery] = [PlannedQuery(
        purpose="broad angle: distinctive title/claim terms",
        query=QuerySpec(keywords=keywords, before_date=before),
    )]
    title_terms = [t for t in _TOKEN_RE.findall(((patent.title if patent else "") or "").lower())
                   if len(t) > 2 and t not in _STOPWORDS][:3]
    if title_terms and len(queries) < n_queries:
        queries.append(PlannedQuery(
            purpose="tight angle: core title terms required",
            query=QuerySpec(keywords=keywords, required_keywords=title_terms,
                            before_date=before),
        ))
    cpc_prefixes = sorted({code[:4] for code in (patent.cpc_codes if patent else [])})[:4]
    if cpc_prefixes and len(queries) < n_queries:
        queries.append(PlannedQuery(
            purpose="classification angle: same CPC art classes",
            query=QuerySpec(keywords=keywords, art_classes=cpc_prefixes,
                            before_date=before),
        ))

    plan = SearchPlan(
        search_type=search_type,
        target=target,
        rationale=("Deterministic pre-run preview of starting angles; the agent "
                   "generates and refines its own queries during execution."),
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

    Agentic model (all constants are rough single-machine figures):

    - **agentic run** (``with_llm_rerank=True``): the cost is dominated by
      provider rounds — expected steps ≈ :data:`BASE_AGENT_STEPS` + 2 per
      planned angle (a search round + an inspect/refine round each), at
      :data:`PER_STEP_LLM_SECONDS` per step plus a per-step tool cost that
      grows logarithmically with corpus size;
    - **degraded run** (no LLM): a single keyword pass per angle, ~0.5 s
      setup plus the same logarithmic index-scan factor;
    - **charting**: ~45 s per claim charted afterwards (one HIGH-effort LLM
      pass per claim across references).

    The estimate is monotonically increasing in every argument, and with the
    default budgets stays within the ≤3-minute envelope the agents enforce.
    """
    n_queries = max(1, len(plan.queries))
    corpus_factor = 0.3 * math.log10(corpus_size + 10)
    if with_llm_rerank:
        expected_steps = BASE_AGENT_STEPS + 2 * n_queries
        seconds = expected_steps * (PER_STEP_LLM_SECONDS + corpus_factor)
    else:
        seconds = 2.0 + n_queries * (0.5 + corpus_factor)
    seconds += 45.0 * max(0, charting_claims)
    return round(seconds, 1)


def humanize_seconds(s: float) -> str:
    """Render a duration for humans: '45 seconds', '3.5 minutes', '1.2 hours'."""
    if s < 90:
        return f"{int(round(s))} seconds"
    if s < 5400:
        return f"{s / 60:.1f} minutes"
    return f"{s / 3600:.1f} hours"


def revise_plan(plan: SearchPlan, feedback: SearchFeedback, llm=None) -> SearchPlan:
    """Revise the pre-run plan preview from user feedback — heuristically.

    No LLM call is made (result-stage feedback is instead injected into the
    resumed agent conversation by the guided loop): queries judged
    ``off_topic`` are dropped, ``too_broad`` queries get a tighter
    ``minimum_match``, ``too_narrow`` ones get it relaxed, and patents
    marked irrelevant are added to the plan's exclusions. ``llm`` is
    accepted for API compatibility and ignored.
    """
    del llm  # plan revision is deterministic; feedback reaches the agent directly
    revised = plan.model_copy(deep=True)
    verdicts = {q.query_index: q.verdict for q in feedback.queries}
    kept: list[PlannedQuery] = []
    for i, planned in enumerate(revised.queries):
        verdict = verdicts.get(i)
        if verdict == "off_topic":
            continue
        if verdict == "too_broad":
            spec = planned.query
            spec.minimum_match = min(len(spec.keywords) or 1,
                                     (spec.minimum_match or
                                      spec.to_search_query().effective_minimum_match()) + 1)
        elif verdict == "too_narrow":
            spec = planned.query
            spec.minimum_match = max(1, (spec.minimum_match or 2) - 1)
        kept.append(planned)
    if kept:
        revised.queries = kept
    _apply_exclusion_feedback(revised, feedback)
    revised.rationale = (revised.rationale + " | revised from user feedback").strip(" |")
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
