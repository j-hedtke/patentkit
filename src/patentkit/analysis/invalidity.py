"""Invalidity analysis: per-reference disclosure assessment and claim charts.

Pipeline (mirrors the production claim-chart flow): split the claim into
atomic limitations, then for each reference assess disclosure of every
limitation (HIGH effort, one call per limitation), optionally attaching
citations via a caller-supplied locator (e.g. built on
:func:`patentkit.parsing.patent_pdf.locate_passage`).
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from patentkit.analysis.claims_analysis import split_atomic_limitations
from patentkit.analysis.prompts import ASSESS_DISCLOSURE
from patentkit.llm.base import LLM, get_llm
from patentkit.models import AtomicLimitation, Patent

logger = logging.getLogger(__name__)

__all__ = [
    "DisclosureFinding",
    "ReferenceChart",
    "ClaimChart",
    "assess_reference",
    "build_claim_chart",
]

DisclosureStatus = Literal["disclosed", "partial", "not_disclosed"]

#: locator callable: verbatim passage -> citation string (e.g. "col. 3, ll. 45-52")
Locator = Callable[[str], Optional[str]]


class DisclosureFinding(BaseModel):
    """Whether one reference discloses one atomic limitation."""

    limitation: AtomicLimitation
    status: DisclosureStatus
    reasoning: str = ""
    quotes: list[str] = Field(default_factory=list)
    citation: Optional[str] = None


class ReferenceChart(BaseModel):
    """All disclosure findings for one prior-art reference."""

    reference_number: str
    reference_title: Optional[str] = None
    findings: list[DisclosureFinding] = Field(default_factory=list)

    def fraction_disclosed(self) -> float:
        """Fraction of limitations this reference fully discloses."""
        if not self.findings:
            return 0.0
        return sum(1 for f in self.findings if f.status == "disclosed") / len(self.findings)


class ClaimChart(BaseModel):
    """A claim chart: one claim's limitations charted against N references."""

    query_patent: str
    claim_number: int
    interpreted_claim: Optional[str] = None
    limitations: list[AtomicLimitation] = Field(default_factory=list)
    references: list[ReferenceChart] = Field(default_factory=list)

    def coverage_summary(self) -> dict[str, float]:
        """Per-reference fraction of limitations with status ``disclosed``."""
        return {ref.reference_number: ref.fraction_disclosed() for ref in self.references}

    def combined_coverage(self) -> float:
        """Fraction of limitations disclosed by at least one reference."""
        if not self.limitations:
            return 0.0
        disclosed: set[str] = set()
        for ref in self.references:
            for finding in ref.findings:
                if finding.status == "disclosed":
                    disclosed.add(finding.limitation.text)
        covered = sum(1 for lim in self.limitations if lim.text in disclosed)
        return covered / len(self.limitations)


def _normalize_status(raw: object) -> DisclosureStatus:
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if value in ("disclosed", "yes", "explicit", "inherent"):
        return "disclosed"
    if value in ("partial", "partially_disclosed", "partial_disclosure"):
        return "partial"
    if value in ("not_disclosed", "no", "none", "undisclosed", "not_found"):
        return "not_disclosed"
    logger.warning("Unrecognized disclosure status %r; treating as not_disclosed", raw)
    return "not_disclosed"


def assess_reference(
    claim_limitations: list[AtomicLimitation],
    reference_text: str,
    reference_number: str,
    reference_title: Optional[str] = None,
    llm: Optional[LLM] = None,
    locator: Optional[Locator] = None,
) -> ReferenceChart:
    """Assess one reference against each limitation (HIGH effort).

    Makes one :data:`~patentkit.analysis.prompts.ASSESS_DISCLOSURE` call per
    limitation. When ``locator`` is given, the first supporting quote of each
    finding is resolved to a citation string (locator failures are logged,
    not raised).
    """
    llm = llm or get_llm("high")
    findings: list[DisclosureFinding] = []
    for limitation in claim_limitations:
        try:
            data = llm.complete_json(
                ASSESS_DISCLOSURE.format(reference=reference_text, limitation=limitation.text),
                max_tokens=4096,
            )
        except Exception:
            logger.error(
                "Disclosure assessment failed for limitation %r against %s",
                limitation.text, reference_number, exc_info=True,
            )
            findings.append(
                DisclosureFinding(
                    limitation=limitation, status="not_disclosed",
                    reasoning="Assessment failed.", quotes=[],
                )
            )
            continue
        if not isinstance(data, dict):
            data = {}
        quotes = [str(q) for q in data.get("quotes", []) if str(q).strip()]
        citation = None
        if locator and quotes:
            try:
                citation = locator(quotes[0])
            except Exception:
                logger.warning("Citation locator failed for %s", reference_number, exc_info=True)
        findings.append(
            DisclosureFinding(
                limitation=limitation,
                status=_normalize_status(data.get("status", "not_disclosed")),
                reasoning=str(data.get("reasoning", "")),
                quotes=quotes,
                citation=citation,
            )
        )
    return ReferenceChart(
        reference_number=reference_number, reference_title=reference_title, findings=findings
    )


def build_claim_chart(
    query_patent: Patent,
    claim_number: int,
    references: list[tuple[str, str]],
    llm: Optional[LLM] = None,
    locator: Optional[Locator] = None,
) -> ClaimChart:
    """Build a claim chart for one claim against several references (HIGH effort).

    Orchestrates: atomic-limitation splitting (reusing precomputed
    ``claim.atomic_limitations`` when present) followed by per-reference
    disclosure assessment.

    Args:
        query_patent: the patent whose claim is being charted.
        claim_number: which claim to chart.
        references: ``(reference_number, reference_text)`` pairs.
        llm: HIGH-effort LLM (defaults via ``get_llm("high")``).
        locator: optional passage -> citation-string callable.
    """
    claim = query_patent.get_claim(claim_number)
    if claim is None:
        raise ValueError(f"Claim {claim_number} not found in {query_patent.patent_number}")
    llm = llm or get_llm("high")

    limitations = claim.atomic_limitations or split_atomic_limitations(claim, query_patent, llm=llm)
    reference_charts = [
        assess_reference(limitations, text, number, llm=llm, locator=locator)
        for number, text in references
    ]
    return ClaimChart(
        query_patent=str(query_patent.patent_number),
        claim_number=claim_number,
        limitations=limitations,
        references=reference_charts,
    )
