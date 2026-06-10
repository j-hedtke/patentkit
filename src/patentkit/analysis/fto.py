"""Freedom-to-operate analysis: a product description against patent claims.

Risk taxonomy follows litigation practice: ``literal`` infringement, ``doe``
(doctrine of equivalents — known interchangeability or function-way-result),
or ``none``; each finding carries a 1-3 confidence and explicit assumptions.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from patentkit.analysis.prompts import FTO_ANALYSIS
from patentkit.llm.base import LLM, get_llm
from patentkit.models import Claim, Patent

logger = logging.getLogger(__name__)

__all__ = ["FtoFinding", "FtoReportData", "analyze_fto"]

FtoRisk = Literal["literal", "doe", "none"]


class FtoFinding(BaseModel):
    """FTO risk assessment of the product against one claim."""

    patent_number: str
    claim_number: int
    risk: FtoRisk
    confidence: int = Field(ge=1, le=3, description="1=low, 2=medium, 3=high")
    assumptions: str = ""
    argument: str = ""


class FtoReportData(BaseModel):
    """Aggregate inputs + findings for an FTO report."""

    product_description: str
    findings: list[FtoFinding] = Field(default_factory=list)
    searched_at: Optional[str] = None  # ISO timestamp of the underlying search
    query_params: dict = Field(default_factory=dict)

    def risk_summary(self) -> dict[str, int]:
        """Count of findings per risk level (always includes all three keys)."""
        summary = {"literal": 0, "doe": 0, "none": 0}
        for finding in self.findings:
            summary[finding.risk] += 1
        return summary


def _normalize_risk(raw: object) -> FtoRisk:
    value = str(raw).strip().lower()
    if value.startswith("literal"):
        return "literal"
    if value.startswith("doe") or "equivalent" in value:
        return "doe"
    if value in ("none", "no infringement", "no_infringement", "no"):
        return "none"
    logger.warning("Unrecognized FTO risk %r; treating as none", raw)
    return "none"


def _clamp_confidence(raw: object) -> int:
    try:
        return min(3, max(1, int(raw)))
    except (TypeError, ValueError):
        return 1


def analyze_fto(
    product_description: str,
    patent: Patent,
    claims: Optional[list[Claim]] = None,
    llm: Optional[LLM] = None,
) -> list[FtoFinding]:
    """Analyze FTO risk of a product against a patent's claims (HIGH effort).

    One :data:`~patentkit.analysis.prompts.FTO_ANALYSIS` call per claim;
    ``claims`` defaults to the patent's independent claims (dependent claims
    cannot be infringed unless their parent is).
    """
    llm = llm or get_llm("high")
    claims = claims if claims is not None else patent.independent_claims
    findings: list[FtoFinding] = []
    for claim in claims:
        try:
            data = llm.complete_json(
                FTO_ANALYSIS.format(
                    product_description=product_description,
                    patent_number=str(patent.patent_number),
                    claim=claim.text,
                ),
                max_tokens=4096,
            )
        except Exception:
            logger.error(
                "FTO analysis failed for %s claim %d",
                patent.patent_number, claim.number, exc_info=True,
            )
            continue
        if not isinstance(data, dict):
            data = {}
        findings.append(
            FtoFinding(
                patent_number=str(patent.patent_number),
                claim_number=claim.number,
                risk=_normalize_risk(data.get("risk", "none")),
                confidence=_clamp_confidence(data.get("confidence", 1)),
                assumptions=str(data.get("assumptions", "")),
                argument=str(data.get("argument", "")),
            )
        )
    return findings
