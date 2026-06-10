"""Canonical patent data model.

Every connector (Google Patents, USPTO ODP/bulk, EPO OPS, PTAB, ...) parses
its source format into these types, and everything downstream — search
stores, analysis skills, formatters, agents — consumes only these types.

Reconciliation: a :class:`Patent` keeps a record of every source it was
assembled from in ``sources``; :meth:`Patent.merge` combines two records of
the same patent, preferring the higher-fidelity source per field and
deduplicating entities (citations, inventors, assignees, classifications).
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

_NUMBER_RE = re.compile(
    r"^(?P<country>[A-Z]{2})?[- ]?(?P<number>[A-Z]?\d[\d,./]*\d|\d)[- ]?(?P<kind>[A-Z]\d?)?$"
)


class PatentNumber(BaseModel):
    """A publication or grant number, normalized across sources."""

    country_code: str = "US"
    number: str
    kind_code: Optional[str] = None

    model_config = {"frozen": True}

    @classmethod
    def parse(cls, raw: str) -> "PatentNumber":
        """Parse e.g. 'US10123456B2', '10,123,456', 'EP1234567A1', 'US 2020/0123456 A1'."""
        cleaned = raw.strip().upper().replace(",", "").replace("/", "").replace(" ", "")
        match = _NUMBER_RE.match(cleaned)
        if not match:
            raise ValueError(f"Cannot parse patent number: {raw!r}")
        return cls(
            country_code=match.group("country") or "US",
            number=match.group("number"),
            kind_code=match.group("kind"),
        )

    def __str__(self) -> str:
        return f"{self.country_code}{self.number}{self.kind_code or ''}"

    def equivalent(self, other: "PatentNumber") -> bool:
        """Same document ignoring kind code (B1 vs B2 etc.)."""
        return self.country_code == other.country_code and self.number.lstrip("0") == other.number.lstrip("0")


class ClaimElement(BaseModel):
    """One node of a claim's element tree (preamble, limitation, sub-element)."""

    text: str
    children: list["ClaimElement"] = Field(default_factory=list)


class AtomicLimitation(BaseModel):
    """The smallest separately-assessable requirement of a claim.

    ``span`` locates the limitation in the claim text as (start, end) character
    offsets, when known.
    """

    text: str
    span: Optional[tuple[int, int]] = None


class Claim(BaseModel):
    number: int
    text: str
    depends_on: Optional[int] = None
    elements: list[ClaimElement] = Field(default_factory=list)
    atomic_limitations: list[AtomicLimitation] = Field(default_factory=list)

    @property
    def is_independent(self) -> bool:
        return self.depends_on is None


class Citation(BaseModel):
    patent_number: PatentNumber
    is_examiner: bool = False
    is_applicant: bool = False
    is_third_party: bool = False
    is_family_to_family: bool = False


class Person(BaseModel):
    name: str
    url: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.name.lower())


class Inventor(Person):
    address: Optional[str] = None


class Assignee(Person):
    pass


class Classification(BaseModel):
    """A CPC/IPC/USPC classification code."""

    scheme: str = "CPC"
    code: str
    description: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.scheme, self.code))


class SpecSection(BaseModel):
    """A heading-delimited section of the specification with numbered lines."""

    heading: Optional[str] = None
    text: str


class SpecChunk(BaseModel):
    """A retrieval-sized chunk of the specification (for RAG stores)."""

    chunk_number: int
    text: str
    section: Optional[str] = None
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None


class Figure(BaseModel):
    url: Optional[str] = None
    page: Optional[int] = None
    caption: Optional[str] = None
    callouts: dict[str, str] = Field(default_factory=dict)  # reference numeral -> part name


class SourceRecord(BaseModel):
    """Provenance for one source a patent record was assembled from."""

    source: str  # e.g. "google_patents", "uspto_odp", "uspto_bulk", "epo_ops"
    fetched_at: Optional[str] = None  # ISO timestamp
    url: Optional[str] = None
    raw_ref: Optional[str] = None  # pointer to raw payload (path, GCS URI, ...)
    fidelity: int = 0  # higher wins on field conflicts


class Patent(BaseModel):
    """Canonical, source-reconciled patent record."""

    patent_number: PatentNumber
    title: Optional[str] = None
    abstract: Optional[str] = None
    specification: Optional[str] = None
    spec_sections: list[SpecSection] = Field(default_factory=list)
    spec_chunks: list[SpecChunk] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    cited_by: list[Citation] = Field(default_factory=list)
    inventors: list[Inventor] = Field(default_factory=list)
    assignees: list[Assignee] = Field(default_factory=list)
    classifications: list[Classification] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    priority_date: Optional[date] = None
    filing_date: Optional[date] = None
    publication_date: Optional[date] = None
    grant_date: Optional[date] = None
    expiration_date: Optional[date] = None
    status: Optional[str] = None
    family: list[PatentNumber] = Field(default_factory=list)
    application_number: Optional[str] = None
    file_wrapper_text: Optional[str] = None
    sources: list[SourceRecord] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def independent_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_independent]

    @property
    def examiner_cited_numbers(self) -> set[str]:
        """Numbers of examiner-cited art — excluded by default in invalidity search."""
        return {str(c.patent_number) for c in self.citations if c.is_examiner}

    @property
    def cpc_codes(self) -> list[str]:
        return [c.code for c in self.classifications if c.scheme == "CPC"]

    def get_claim(self, number: int) -> Optional[Claim]:
        return next((c for c in self.claims if c.number == number), None)

    def best_effective_date(self) -> Optional[date]:
        """Earliest of priority/filing date — the cutoff for prior art."""
        candidates = [d for d in (self.priority_date, self.filing_date) if d]
        return min(candidates) if candidates else None

    def text_for_search(self) -> str:
        parts = [self.title or "", self.abstract or ""]
        parts += [c.text for c in self.claims]
        parts.append(self.specification or "")
        return "\n".join(p for p in parts if p)

    def merge(self, other: "Patent") -> "Patent":
        """Reconcile this record with another record of the same patent.

        Scalar fields: the value from the higher-fidelity source wins; a
        present value always beats ``None``. List fields: union with
        entity-level dedup. Sources: concatenated provenance.
        """
        if not self.patent_number.equivalent(other.patent_number):
            raise ValueError(
                f"Refusing to merge different patents: {self.patent_number} vs {other.patent_number}"
            )
        mine = max((s.fidelity for s in self.sources), default=0)
        theirs = max((s.fidelity for s in other.sources), default=0)
        primary, secondary = (self, other) if mine >= theirs else (other, self)

        merged = primary.model_copy(deep=True)
        for field_name in (
            "title", "abstract", "specification", "priority_date", "filing_date",
            "publication_date", "grant_date", "expiration_date", "status",
            "application_number", "file_wrapper_text",
        ):
            if getattr(merged, field_name) is None:
                setattr(merged, field_name, getattr(secondary, field_name))
        if not merged.claims:
            merged.claims = [c.model_copy(deep=True) for c in secondary.claims]
        if not merged.spec_sections:
            merged.spec_sections = list(secondary.spec_sections)
        if not merged.spec_chunks:
            merged.spec_chunks = list(secondary.spec_chunks)
        if not merged.figures:
            merged.figures = list(secondary.figures)

        merged.citations = _dedup_citations(primary.citations + secondary.citations)
        merged.cited_by = _dedup_citations(primary.cited_by + secondary.cited_by)
        merged.inventors = _dedup_by(primary.inventors + secondary.inventors, lambda p: p.name.lower())
        merged.assignees = _dedup_by(primary.assignees + secondary.assignees, lambda p: p.name.lower())
        merged.classifications = _dedup_by(
            primary.classifications + secondary.classifications, lambda c: (c.scheme, c.code)
        )
        merged.family = _dedup_by(primary.family + secondary.family, str)
        merged.sources = list(primary.sources) + list(secondary.sources)
        merged.extras = {**secondary.extras, **primary.extras}
        return merged


def _dedup_by(items: list, key) -> list:
    seen: set = set()
    out = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _dedup_citations(citations: list[Citation]) -> list[Citation]:
    """Dedup by patent number, OR-ing the origin flags."""
    by_number: dict[str, Citation] = {}
    for cit in citations:
        key = str(cit.patent_number)
        if key in by_number:
            prev = by_number[key]
            by_number[key] = Citation(
                patent_number=cit.patent_number,
                is_examiner=prev.is_examiner or cit.is_examiner,
                is_applicant=prev.is_applicant or cit.is_applicant,
                is_third_party=prev.is_third_party or cit.is_third_party,
                is_family_to_family=prev.is_family_to_family or cit.is_family_to_family,
            )
        else:
            by_number[key] = cit
    return list(by_number.values())
