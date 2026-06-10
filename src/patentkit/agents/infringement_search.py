"""Infringement (evidence-of-use) search agent.

Ranks candidate products against the independent-claim limitations of a
patent. Candidates are duck-typed: plain dicts with ``name`` /
``description`` / ``url`` keys, or connector ``Product`` objects exposing
the same attributes (e.g. from ``patentkit.connectors``). Scoring uses one
batched HIGH-effort LLM JSON call when an LLM is configured; otherwise a
token-overlap heuristic, so the agent works keys-free.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, Field

from patentkit.agents._support import report_progress
from patentkit.agents.planner import _STOPWORDS
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
    #: ranked candidates: {name, description, url, score, rationale}
    results: list[dict] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)


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
    """Keys-free fallback: fraction of distinctive limitation tokens present, 0-10."""
    if not limitation_tokens:
        return 0.0
    tokens = set(_TOKEN_RE.findall(text.lower()))
    return 10.0 * len(tokens & limitation_tokens) / len(limitation_tokens)


class InfringementSearchAgent:
    """Ranks candidate products against a patent's claim limitations.

    Args:
        llm: optional :class:`patentkit.llm.LLM` for batched HIGH-effort
            candidate scoring; ``None`` falls back to token overlap.
        catalogs: optional product catalogs consulted when no explicit
            ``product_candidates`` are passed. Each catalog may be a plain
            list of candidates or an object exposing a ``products`` attribute.
    """

    def __init__(self, llm=None, catalogs: Optional[list] = None):
        self.llm = llm
        self.catalogs = list(catalogs or [])

    def _gather_candidates(self, product_candidates: Optional[Sequence[Any]]) -> list[dict]:
        raw: list[Any] = list(product_candidates or [])
        if not raw:
            for catalog in self.catalogs:
                items = catalog if isinstance(catalog, (list, tuple)) else getattr(catalog, "products", [])
                raw += list(items)
        return [p for p in (_as_product(c) for c in raw) if p["name"] or p["description"]]

    def _llm_rank(self, limitations: list[str], products: list[dict],
                  evidence_texts: Optional[Sequence[str]]) -> Optional[list[dict]]:
        """One batched JSON call scoring all candidates; None on failure."""
        if self.llm is None or not products:
            return None
        lines = []
        for i, product in enumerate(products):
            evidence = ""
            if evidence_texts and i < len(evidence_texts):
                evidence = f' evidence: "{str(evidence_texts[i])[:400]}"'
            lines.append(f'- name: "{product["name"]}" description: '
                         f'"{product["description"][:500]}"{evidence}')
        prompt = (
            "Claim limitations:\n" + "\n".join(f"{i + 1}. {t[:400]}" for i, t in enumerate(limitations[:20]))
            + "\n\nScore each product 0-10 on how likely it practices ALL of the "
              "limitations above, with a one-sentence rationale. Respond with ONLY "
              'a JSON array [{"name": "...", "score": <0-10>, "rationale": "..."}].\n\n'
              "Products:\n" + "\n".join(lines)
        )
        try:
            raw = self.llm.complete_json(prompt, max_tokens=4096)
            if isinstance(raw, dict):
                raw = raw.get("results", raw.get("products", []))
            scored = {}
            for item in raw if isinstance(raw, list) else []:
                if isinstance(item, dict) and "name" in item:
                    try:
                        score = max(0.0, min(10.0, float(item.get("score", 0))))
                    except (TypeError, ValueError):
                        continue
                    scored[str(item["name"])] = (score, str(item.get("rationale", "")))
            if not scored:
                return None
            out = []
            for product in products:
                score, rationale = scored.get(product["name"], (0.0, "not scored by LLM"))
                out.append({**product, "score": round(score, 4), "rationale": rationale})
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM infringement scoring failed (%s); using token overlap", exc)
            return None

    def search(
        self,
        patent: Patent,
        claims: Optional[list[int]] = None,
        product_candidates: Optional[Sequence[Any]] = None,
        evidence_texts: Optional[Sequence[str]] = None,
        final_k: int = 10,
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
                by index) added to the LLM context.
            final_k: candidates returned.
            progress: optional callback receiving stage updates.
        """
        t0 = time.monotonic()
        limitations = _limitations(patent, claims)
        products = self._gather_candidates(product_candidates)
        report_progress(progress, f"infringement: ranking {len(products)} candidates "
                                  f"against {len(limitations)} limitations")

        ranked = self._llm_rank(limitations, products, evidence_texts)
        if ranked is None:  # keys-free token-overlap fallback
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
                    "rationale": "token-overlap heuristic (no LLM configured)",
                })
        ranked.sort(key=lambda item: item["score"], reverse=True)

        claim_numbers = claims or [c.number for c in (patent.independent_claims or patent.claims)]
        return InfringementSearchResult(
            target=str(patent.patent_number),
            claims=claim_numbers,
            results=ranked[:final_k],
            timing={"total": round(time.monotonic() - t0, 4)},
        )


__all__ = ["InfringementSearchAgent", "InfringementSearchResult"]
