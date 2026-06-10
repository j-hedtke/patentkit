"""Infringement evidence analysis: product evidence vs. claim limitations.

Each limitation of the target claim is assessed against a concatenated,
source-tagged evidence corpus (product pages, datasheets, teardowns, ...),
yielding per-limitation findings with verbatim evidence quotes.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from patentkit.analysis.claims_analysis import split_atomic_limitations
from patentkit.analysis.prompts import INFRINGEMENT_ANALYSIS
from patentkit.llm.base import LLM, get_llm
from patentkit.models import AtomicLimitation, Patent

logger = logging.getLogger(__name__)

__all__ = ["EvidenceItem", "InfringementFinding", "analyze_infringement"]

InfringementStatus = Literal["met", "likely", "unclear", "not_met"]


class EvidenceItem(BaseModel):
    """One piece of supporting evidence for an infringement finding."""

    source_url: str = ""
    quote: str = ""
    note: str = ""


class InfringementFinding(BaseModel):
    """Whether the product evidence shows one limitation is met."""

    limitation: AtomicLimitation
    status: InfringementStatus
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reasoning: str = ""


def _normalize_status(raw: object) -> InfringementStatus:
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if value in ("met", "yes", "satisfied", "present"):
        return "met"
    if value in ("likely", "probably_met", "likely_met"):
        return "likely"
    if value in ("not_met", "no", "absent", "not_present"):
        return "not_met"
    if value in ("unclear", "unknown", "insufficient", "indeterminate"):
        return "unclear"
    logger.warning("Unrecognized infringement status %r; treating as unclear", raw)
    return "unclear"


def analyze_infringement(
    patent: Patent,
    claim_number: int,
    product_name: str,
    evidence_texts: list[tuple[str, str]],
    llm: Optional[LLM] = None,
) -> list[InfringementFinding]:
    """Analyze product evidence against each limitation of a claim (HIGH effort).

    Args:
        patent: the asserted patent.
        claim_number: which claim to analyze.
        product_name: name of the accused product.
        evidence_texts: ``(source, text)`` pairs; concatenated into one
            corpus with ``<evidence source="...">`` tags so the model can
            attribute quotes.
        llm: HIGH-effort LLM (defaults via ``get_llm("high")``).

    One :data:`~patentkit.analysis.prompts.INFRINGEMENT_ANALYSIS` call per
    limitation. Limitations come from ``claim.atomic_limitations`` when
    precomputed, otherwise from :func:`split_atomic_limitations`.
    """
    claim = patent.get_claim(claim_number)
    if claim is None:
        raise ValueError(f"Claim {claim_number} not found in {patent.patent_number}")
    llm = llm or get_llm("high")

    limitations: list[AtomicLimitation] = (
        claim.atomic_limitations or split_atomic_limitations(claim, patent, llm=llm)
    )
    corpus = "\n\n".join(
        f'<evidence source="{source}">\n{text}\n</evidence>' for source, text in evidence_texts
    )

    findings: list[InfringementFinding] = []
    for limitation in limitations:
        try:
            data = llm.complete_json(
                INFRINGEMENT_ANALYSIS.format(
                    product_name=product_name, limitation=limitation.text, evidence=corpus
                ),
                max_tokens=4096,
            )
        except Exception:
            logger.error(
                "Infringement analysis failed for limitation %r", limitation.text, exc_info=True
            )
            findings.append(
                InfringementFinding(
                    limitation=limitation, status="unclear",
                    reasoning="Analysis failed.", evidence=[],
                )
            )
            continue
        if not isinstance(data, dict):
            data = {}
        evidence = []
        for item in data.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            evidence.append(
                EvidenceItem(
                    source_url=str(item.get("source", item.get("source_url", ""))),
                    quote=str(item.get("quote", "")),
                    note=str(item.get("note", "")),
                )
            )
        findings.append(
            InfringementFinding(
                limitation=limitation,
                status=_normalize_status(data.get("status", "unclear")),
                evidence=evidence,
                reasoning=str(data.get("reasoning", "")),
            )
        )
    return findings
