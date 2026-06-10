"""Invalidity claim-charting agent.

Wraps ``patentkit.analysis.invalidity.build_claim_chart`` (lazily imported)
to chart one or more claims of a query patent against a set of references,
aggregate per-claim coverage, and render markdown — plus an optional .docx
via ``patentkit.formatting`` when that extra is installed.

Example::

    from patentkit.agents import InvalidityChartingAgent
    from patentkit.llm import get_llm

    agent = InvalidityChartingAgent(llm=get_llm("high"))
    out = agent.chart(query_patent, references=[("US1234567B2", ref_text)],
                      claims=[1], out_docx="chart.docx")
    print(out.markdown)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from patentkit.agents._support import report_progress
from patentkit.models import Patent

logger = logging.getLogger(__name__)


class ChartingResult(BaseModel):
    """Serializable outcome of a charting run."""

    target: str
    claims: list[int] = Field(default_factory=list)
    #: per-claim chart data (serialized ClaimChart objects)
    charts: list[dict] = Field(default_factory=list)
    #: claim number (as str) -> coverage summary
    coverage: dict[str, Any] = Field(default_factory=dict)
    markdown: str = ""
    docx_path: Optional[str] = None
    error: Optional[str] = None
    timing: dict[str, float] = Field(default_factory=dict)


def _to_jsonable(obj: Any) -> Any:
    """Best-effort serialization of a ClaimChart-like object."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        return {k: str(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return {"repr": repr(obj)}


def _local_markdown(target: str, charts: list[dict], coverage: dict[str, Any]) -> str:
    """Minimal markdown rendering used when patentkit.formatting is absent."""
    lines = [f"# Invalidity claim charts — {target}", ""]
    for chart in charts:
        claim = chart.get("claim_number", "?")
        lines.append(f"## Claim {claim}")
        lines.append("")
        lines.append(f"Coverage: {coverage.get(str(claim), 'n/a')}")
        lines.append("")
        lines.append("```json-ish")
        lines.append(str(chart)[:4000])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


class InvalidityChartingAgent:
    """Builds element-by-element invalidity claim charts.

    Args:
        llm: optional :class:`patentkit.llm.LLM` passed through to
            ``build_claim_chart`` (HIGH-effort recommended).
    """

    def __init__(self, llm=None):
        self.llm = llm

    def chart(
        self,
        query_patent: Patent,
        references: list[tuple[str, str]],
        claims: Optional[list[int]] = None,
        out_docx: Optional[str] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> ChartingResult:
        """Chart each selected claim against the references.

        Args:
            query_patent: the patent whose claims are charted.
            references: ``(patent_number, reference_text_or_title)`` tuples
                passed through to ``build_claim_chart``.
            claims: claim numbers to chart (default: independent claims).
            out_docx: when given, attempt a .docx rendering via
                ``patentkit.formatting.claim_chart`` (skipped, markdown-only,
                if that module or the docx extra is unavailable).
            progress: optional callback receiving per-claim updates.
        """
        t0 = time.monotonic()
        claims = claims or [c.number for c in query_patent.independent_claims] or [1]
        target = str(query_patent.patent_number)

        try:
            from patentkit.analysis.invalidity import build_claim_chart  # noqa: PLC0415 — lazy by design
        except ImportError:
            message = ("patentkit.analysis.invalidity is unavailable — install/enable the "
                       "analysis module to build claim charts.")
            logger.warning(message)
            return ChartingResult(target=target, claims=claims, markdown=message,
                                  error=message, timing={"total": round(time.monotonic() - t0, 4)})

        charts: list[dict] = []
        coverage: dict[str, Any] = {}
        for claim_number in claims:
            report_progress(progress, f"charting claim {claim_number} against {len(references)} references")
            try:
                chart = build_claim_chart(query_patent, claim_number, references, self.llm)
            except Exception as exc:  # noqa: BLE001 — keep charting remaining claims
                logger.warning("charting claim %s failed: %s", claim_number, exc)
                coverage[str(claim_number)] = f"error: {exc}"
                continue
            data = _to_jsonable(chart)
            data.setdefault("claim_number", claim_number)
            charts.append(data)
            try:
                coverage[str(claim_number)] = chart.coverage_summary()
            except Exception as exc:  # noqa: BLE001
                coverage[str(claim_number)] = f"coverage unavailable: {exc}"

        markdown = self._render_markdown(target, charts, coverage)
        docx_path = self._render_docx(charts, out_docx) if out_docx else None

        return ChartingResult(
            target=target, claims=claims, charts=charts, coverage=coverage,
            markdown=markdown, docx_path=docx_path,
            timing={"total": round(time.monotonic() - t0, 4)},
        )

    def _render_markdown(self, target: str, charts: list[dict], coverage: dict[str, Any]) -> str:
        try:
            from patentkit.formatting.claim_chart import render_markdown  # noqa: PLC0415
            return render_markdown(charts)
        except Exception:  # noqa: BLE001 — module absent or different API
            return _local_markdown(target, charts, coverage)

    def _render_docx(self, charts: list[dict], out_docx: str) -> Optional[str]:
        try:
            from patentkit.formatting.claim_chart import render_docx  # noqa: PLC0415
            render_docx(charts, out_docx)
            return out_docx
        except ImportError:
            logger.warning("patentkit.formatting / docx extra unavailable; returning markdown only")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("docx rendering failed: %s", exc)
            return None


__all__ = ["InvalidityChartingAgent", "ChartingResult"]
