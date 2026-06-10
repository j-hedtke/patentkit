"""User-feedback primitives for the guided search loop.

These models capture structured human feedback on a search plan, on
individual results, and on individual passages, so an agent (or the
:class:`~patentkit.agents.guided.GuidedSearch` loop) can revise its plan.
All models are plain pydantic, JSON-serializable, and safe to round-trip
through MCP / OpenAI tool calls.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ResultFeedback(BaseModel):
    """Feedback on one ranked result (a patent)."""

    patent_number: str
    #: True = relevant, False = irrelevant, None = unjudged
    relevant: Optional[bool] = None
    note: Optional[str] = None


class PassageFeedback(BaseModel):
    """Feedback on one highlighted passage of a result."""

    patent_number: str
    passage_text: str
    relevant: Optional[bool] = None
    note: Optional[str] = None


class QueryFeedback(BaseModel):
    """A verdict on one planned query (indexed into ``SearchPlan.queries``)."""

    query_index: int
    verdict: Literal["good", "too_broad", "too_narrow", "off_topic"]
    note: Optional[str] = None


class SearchFeedback(BaseModel):
    """A full bundle of user feedback collected in one guided-loop turn."""

    results: list[ResultFeedback] = Field(default_factory=list)
    passages: list[PassageFeedback] = Field(default_factory=list)
    queries: list[QueryFeedback] = Field(default_factory=list)
    free_text: Optional[str] = None

    def summary_for_llm(self) -> str:
        """Render the feedback as a compact plain-text block for an LLM prompt."""
        lines: list[str] = []
        if self.queries:
            lines.append("Query feedback:")
            for q in self.queries:
                lines.append(f"- query #{q.query_index}: {q.verdict}" + (f" ({q.note})" if q.note else ""))
        relevant = [r for r in self.results if r.relevant is True]
        irrelevant = [r for r in self.results if r.relevant is False]
        if relevant:
            lines.append("Results marked RELEVANT: " + ", ".join(r.patent_number for r in relevant))
        if irrelevant:
            lines.append("Results marked IRRELEVANT: " + ", ".join(r.patent_number for r in irrelevant))
        for r in self.results:
            if r.note:
                lines.append(f"- note on {r.patent_number}: {r.note}")
        for p in self.passages:
            verdict = {True: "relevant", False: "irrelevant", None: "noted"}[p.relevant]
            snippet = p.passage_text[:120].replace("\n", " ")
            line = f"- passage of {p.patent_number} judged {verdict}: \"{snippet}\""
            if p.note:
                line += f" ({p.note})"
            lines.append(line)
        if self.free_text:
            lines.append(f"Free-text feedback: {self.free_text}")
        return "\n".join(lines) if lines else "(no feedback given)"


__all__ = ["ResultFeedback", "PassageFeedback", "QueryFeedback", "SearchFeedback"]
