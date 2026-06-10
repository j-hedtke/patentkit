"""Analysis skills: claim interpretation/splitting, invalidity claim charts,
FTO, infringement, drafting, and keyword generation.

Every LLM-backed function accepts ``llm: LLM | None = None`` and defaults via
``get_llm`` at the documented effort tier (LOW extraction/formatting, MEDIUM
interpretation/passage selection, HIGH assessment/charting/drafting).
"""

from patentkit.analysis.claims_analysis import (
    generate_keywords,
    interpret_claim,
    split_atomic_limitations,
)
from patentkit.analysis.drafting import (
    check_antecedent_basis,
    draft_claims,
    draft_spec_section,
)
from patentkit.analysis.fto import FtoFinding, FtoReportData, analyze_fto
from patentkit.analysis.infringement import (
    EvidenceItem,
    InfringementFinding,
    analyze_infringement,
)
from patentkit.analysis.invalidity import (
    ClaimChart,
    DisclosureFinding,
    ReferenceChart,
    assess_reference,
    build_claim_chart,
)

__all__ = [
    "generate_keywords",
    "interpret_claim",
    "split_atomic_limitations",
    "check_antecedent_basis",
    "draft_claims",
    "draft_spec_section",
    "FtoFinding",
    "FtoReportData",
    "analyze_fto",
    "EvidenceItem",
    "InfringementFinding",
    "analyze_infringement",
    "ClaimChart",
    "DisclosureFinding",
    "ReferenceChart",
    "assess_reference",
    "build_claim_chart",
]
