"""Plug-and-play search store interfaces.

Two store families, both consuming the canonical :class:`~patentkit.models.Patent`:

- :class:`KeywordStore` — traditional keyword / BM25 indexes (in-memory BM25,
  Elasticsearch). Queried with a :class:`SearchQuery`.
- :class:`VectorStore` — embedding / RAG stores over specification chunks and
  claims (in-memory, Elasticsearch dense-vector). Queried with text or vectors.

Both return :class:`SearchResult` lists so rankers, agents, and formatters are
backend-agnostic. :class:`HybridSearcher` (search/hybrid.py) fuses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional, Protocol, runtime_checkable

from patentkit.models import Patent, PatentNumber


@dataclass
class SearchQuery:
    """The full query parameter set exposed to users and agents.

    Any field may be omitted; backends apply what they support and must
    document what they ignore.
    """

    keywords: list[str] = field(default_factory=list)
    #: keywords that MUST appear (AND semantics)
    required_keywords: list[str] = field(default_factory=list)
    #: tokens/phrases that must NOT appear
    excluded_keywords: list[str] = field(default_factory=list)
    #: free-text query (used by phrase/semantic matching)
    text: Optional[str] = None
    #: minimum number of ``keywords`` that must match (default: len//3, >=1)
    minimum_match: Optional[int] = None
    #: fields to search; subset of {"title", "abstract", "claims", "specification"}
    fields: list[str] = field(default_factory=lambda: ["title", "abstract", "claims", "specification"])
    #: CPC/IPC art class prefixes, e.g. ["G06F16", "H04L"]
    art_classes: list[str] = field(default_factory=list)
    inventors: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    #: only documents effective before this date (prior-art cutoff)
    before_date: Optional[date] = None
    after_date: Optional[date] = None
    countries: list[str] = field(default_factory=list)
    #: explicit allow/deny lists of patent numbers
    include_numbers: list[PatentNumber] = field(default_factory=list)
    exclude_numbers: list[PatentNumber] = field(default_factory=list)
    limit: int = 100

    def effective_minimum_match(self) -> int:
        if self.minimum_match is not None:
            return self.minimum_match
        return max(1, len(self.keywords) // 3)


@dataclass
class Passage:
    """A highlighted passage supporting a result's relevance."""

    text: str
    field: str = "specification"
    score: float = 0.0
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class SearchResult:
    patent_number: PatentNumber
    score: float
    patent: Optional[Patent] = None
    passages: list[Passage] = field(default_factory=list)
    #: backend-specific explanation of the score
    explanation: Optional[str] = None

    @property
    def title(self) -> str | None:
        return self.patent.title if self.patent else None


@runtime_checkable
class KeywordStore(Protocol):
    def index(self, patents: Iterable[Patent]) -> int:
        """Add/update patents; returns count indexed."""
        ...

    def search(self, query: SearchQuery) -> list[SearchResult]: ...

    def get(self, number: PatentNumber) -> Optional[Patent]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class VectorStore(Protocol):
    def index(self, patents: Iterable[Patent]) -> int:
        """Chunk + embed + store patents; returns count indexed."""
        ...

    def search_text(self, text: str, *, limit: int = 50,
                    query: SearchQuery | None = None) -> list[SearchResult]:
        """Embed ``text`` and run nearest-neighbor search; ``query`` supplies
        metadata filters (dates, classes, exclusions)."""
        ...

    def get(self, number: PatentNumber) -> Optional[Patent]: ...


class EmbeddingProvider(Protocol):
    """Pluggable embedding backend (OpenAI, Voyage, local)."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def apply_metadata_filters(patent: Patent, query: SearchQuery) -> bool:
    """Shared pure-Python filter used by in-memory backends. True = keep."""
    number = patent.patent_number
    if any(number.equivalent(n) for n in query.exclude_numbers):
        return False
    if query.include_numbers and not any(number.equivalent(n) for n in query.include_numbers):
        return False
    if query.countries and number.country_code not in query.countries:
        return False
    if query.before_date:
        effective = patent.best_effective_date() or patent.publication_date
        if effective is None or effective >= query.before_date:
            return False
    if query.after_date:
        effective = patent.best_effective_date() or patent.publication_date
        if effective is None or effective <= query.after_date:
            return False
    if query.art_classes:
        codes = [c.code for c in patent.classifications]
        if not any(code.startswith(prefix) for prefix in query.art_classes for code in codes):
            return False
    if query.inventors:
        names = {i.name.lower() for i in patent.inventors}
        if not any(q.lower() in name for q in query.inventors for name in names):
            return False
    if query.assignees:
        names = {a.name.lower() for a in patent.assignees}
        if not any(q.lower() in name for q in query.assignees for name in names):
            return False
    if query.excluded_keywords:
        text = patent.text_for_search().lower()
        if any(tok.lower() in text for tok in query.excluded_keywords):
            return False
    return True
