"""Agentic infringement (evidence-of-use) search.

Ranks candidate products against the independent-claim limitations of a
patent. With an LLM configured, an agent conversation (:class:`~patentkit.
agents.agentic.AgenticSearchRunner`) reasons over the limitations, the
candidate descriptions, and any evidence texts, may re-read the target via
its tools, and finishes with a ranked candidate list (full saved trace,
resumable conversation). Candidates are duck-typed: plain dicts with
``name`` / ``description`` / ``url`` keys, or connector ``Product`` objects
exposing the same attributes.

Without an LLM the agent degrades to a token-overlap heuristic (clearly
labeled), so it works keys-free.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.agents._support import report_progress
from patentkit.agents.agentic import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_STEPS,
    AgenticSearchRunner,
    SearchTrace,
)
from patentkit.agents.invalidity_search import step_summary
from patentkit.agents.planner import _STOPWORDS
from patentkit.llm.tools import TraceStep
from patentkit.models import Claim, Patent

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class InfringementSearchResult(BaseModel):
    """Serializable outcome of an infringement candidate ranking.

    Results identify products *worth investigating* — they are not an
    infringement opinion; element-by-element analysis by counsel is required.
    """

    target: str
    claims: list[int] = Field(default_factory=list)
    #: ranked candidates, best first:
    #: {name, description, url, score (=confidence 0-1), rationale}
    results: list[dict] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    trace: Optional[SearchTrace] = None
    stop_reason: Optional[str] = None
    #: neutral-schema agent conversation for resumption (agentic mode)
    conversation: Optional[list[dict]] = None


def _as_product(candidate: Any) -> dict:
    """Normalize a dict or duck-typed Product object to {name, description, url}."""
    if isinstance(candidate, dict):
        return {
            "name": str(candidate.get("name", "") or candidate.get("title", "")),
            "description": str(candidate.get("description", "") or candidate.get("text", "")),
            "url": candidate.get("url"),
        }
    return {
        "name": str(getattr(candidate, "name", "") or ""),
        "description": str(getattr(candidate, "description", "") or ""),
        "url": getattr(candidate, "url", None),
    }


def _limitations(patent: Patent, claims: Optional[list[int]]) -> list[str]:
    """Limitation texts of the selected (default: independent) claims."""
    selected: list[Claim]
    if claims:
        selected = [c for c in patent.claims if c.number in set(claims)]
    else:
        selected = patent.independent_claims or patent.claims
    texts: list[str] = []
    for claim in selected:
        if claim.atomic_limitations:
            texts += [lim.text for lim in claim.atomic_limitations]
        else:
            texts.append(claim.text)
    return texts


def _overlap_score(text: str, limitation_tokens: set[str]) -> float:
    """Keys-free fallback: fraction (0-1) of distinctive limitation tokens present."""
    if not limitation_tokens:
        return 0.0
    tokens = set(_TOKEN_RE.findall(text.lower()))
    return len(tokens & limitation_tokens) / len(limitation_tokens)


class InfringementSearchAgent:
    """Ranks candidate products against a patent's claim limitations.

    Args:
        llm: optional :class:`patentkit.llm.LLM` driving the agentic
            ranking; ``None`` falls back to token overlap.
        catalogs: optional product catalogs consulted when no explicit
            ``product_candidates`` are passed. Each catalog may be a plain
            list of candidates or an object exposing a ``products`` attribute.
        keyword_store: optional patent store the agent may consult for
            context (default: a throwaway in-memory store holding only the
            target patent).
    """

    def __init__(self, llm=None, catalogs: Optional[list] = None, keyword_store=None):
        self.llm = llm
        self.catalogs = list(catalogs or [])
        self.keyword_store = keyword_store

    def _gather_candidates(self, product_candidates: Optional[Sequence[Any]]) -> list[dict]:
        raw: list[Any] = list(product_candidates or [])
        if not raw:
            for catalog in self.catalogs:
                items = catalog if isinstance(catalog, (list, tuple)) else getattr(catalog, "products", [])
                raw += list(items)
        return [p for p in (_as_product(c) for c in raw) if p["name"] or p["description"]]

    def _store_for(self, patent: Patent):
        """The agent's patent store: configured one, or a tiny in-memory
        store holding the target so get_patent works."""
        if self.keyword_store is not None:
            return self.keyword_store
        from patentkit.search.bm25 import BM25Store  # noqa: PLC0415 — cheap, optional path
        store = BM25Store()
        store.index([patent])
        return store

    def search(
        self,
        patent: Patent,
        claims: Optional[list[int]] = None,
        product_candidates: Optional[Sequence[Any]] = None,
        evidence_texts: Optional[Sequence[str]] = None,
        final_k: int = 10,
        *,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Optional[Callable[[TraceStep], None]] = None,
        feedback_messages: Sequence[str] = (),
        resume_messages: Optional[list[dict]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> InfringementSearchResult:
        """Rank candidate products by likelihood of practicing the claims.

        Args:
            patent: the asserted patent.
            claims: claim numbers in focus (default: independent claims).
            product_candidates: dicts or Product-like objects
                (``.name`` / ``.description`` / ``.url``); falls back to the
                constructor's catalogs when omitted.
            evidence_texts: optional per-candidate evidence strings (aligned
                by index) added to the agent's context.
            final_k: candidates returned.
            budget_seconds / max_steps: agent budgets (agentic mode).
            on_step: live callback receiving every trace step.
            feedback_messages / resume_messages: continue a previous agent
                conversation with injected user feedback.
            progress: optional callback receiving human-readable updates.
        """
        t0 = time.monotonic()
        limitations = _limitations(patent, claims)
        products = self._gather_candidates(product_candidates)
        report_progress(progress, f"infringement: ranking {len(products)} candidates "
                                  f"against {len(limitations)} limitations")
        claim_numbers = claims or [c.number for c in (patent.independent_claims or patent.claims)]

        if self.llm is None or not products:
            return self._degraded_search(patent, claim_numbers, limitations, products,
                                         evidence_texts, final_k, t0=t0)

        def _step(step: TraceStep) -> None:
            report_progress(progress, step_summary(step))
            if on_step is not None:
                on_step(step)

        evidence = []
        for i, product in enumerate(products):
            entry = dict(product)
            if evidence_texts and i < len(evidence_texts):
                entry["evidence"] = str(evidence_texts[i])
            evidence.append(entry)

        runner = AgenticSearchRunner(self._store_for(patent), None, self.llm,
                                     max_steps=max_steps, budget_seconds=budget_seconds)
        outcome = runner.run(
            "infringement",
            target_patent=patent,
            claims=claim_numbers,
            evidence=evidence,
            exclusions={},
            final_k=final_k,
            feedback_messages=feedback_messages,
            resume_messages=resume_messages,
            on_step=_step,
        )

        by_name = {p["name"]: p for p in products}
        ranked: list[dict] = []
        for candidate in outcome.results:
            product = by_name.get(candidate.number,
                                  {"name": candidate.number, "description": "", "url": None})
            ranked.append({**product, "score": round(candidate.confidence, 4),
                           "rationale": candidate.why})
        return InfringementSearchResult(
            target=str(patent.patent_number),
            claims=claim_numbers,
            results=ranked[:final_k],
            timing={"total": round(time.monotonic() - t0, 4), "agent": outcome.elapsed_s},
            trace=outcome.trace,
            stop_reason=outcome.stop_reason,
            conversation=outcome.messages,
        )

    # ------------------------------------------------------- degraded mode
    def _degraded_search(self, patent: Patent, claim_numbers: list[int],
                         limitations: list[str], products: list[dict],
                         evidence_texts: Optional[Sequence[str]], final_k: int, *,
                         t0: float) -> InfringementSearchResult:
        """Keys-free fallback: token-overlap heuristic, clearly labeled."""
        limitation_tokens = {
            t for text in limitations for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 2 and t not in _STOPWORDS
        }
        ranked = []
        for i, product in enumerate(products):
            text = product["name"] + " " + product["description"]
            if evidence_texts and i < len(evidence_texts):
                text += " " + str(evidence_texts[i])
            ranked.append({
                **product,
                "score": round(_overlap_score(text, limitation_tokens), 4),
                "rationale": "degraded mode: token-overlap heuristic (no LLM configured)",
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return InfringementSearchResult(
            target=str(patent.patent_number),
            claims=claim_numbers,
            results=ranked[:final_k],
            timing={"total": round(time.monotonic() - t0, 4)},
            trace=None,
            stop_reason="degraded",
        )


__all__ = ["InfringementSearchAgent", "InfringementSearchResult"]
