"""Claim-level analysis skills: interpretation, atomic-limitation splitting,
and search-keyword generation.

Every function accepts ``llm: LLM | None`` and defaults via
:func:`patentkit.llm.base.get_llm` at the documented effort tier.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

from patentkit.analysis.prompts import (
    INTERPRET_CLAIM,
    KEYWORD_GENERATION,
    MAP_LIMITATION_SPANS,
    SPLIT_ATOMIC_LIMITATIONS,
)
from patentkit.llm.base import LLM, get_llm
from patentkit.models import AtomicLimitation, Claim, Patent

logger = logging.getLogger(__name__)

__all__ = ["interpret_claim", "split_atomic_limitations", "generate_keywords"]


def interpret_claim(claim: Claim, patent: Patent, llm: Optional[LLM] = None) -> str:
    """Interpret ``claim`` in light of its patent's specification (MEDIUM effort).

    Returns the claim text with inline parenthetical annotations where the
    specification clearly defines a term; unannotated terms carry their plain
    and ordinary meaning.
    """
    llm = llm or get_llm("medium")
    specification = patent.specification or patent.abstract or ""
    prompt = INTERPRET_CLAIM.format(specification=specification, claim=claim.text)
    return llm.complete(prompt, max_tokens=4096).text.strip()


def split_atomic_limitations(
    claim: Claim, patent: Optional[Patent] = None, llm: Optional[LLM] = None
) -> list[AtomicLimitation]:
    """Split ``claim`` into atomic limitations with character spans (HIGH effort).

    Two LLM calls: one to split the claim into limitations, one to map each
    limitation to a (start, end) character span in the claim text. Spans are
    validated against ``claim.text``; invalid or missing spans fall back to a
    case-insensitive substring search (span ``None`` when that fails too).
    ``patent`` is currently unused beyond logging context but kept in the
    signature so callers can supply specification context later.
    """
    llm = llm or get_llm("high")
    raw = llm.complete_json(SPLIT_ATOMIC_LIMITATIONS.format(claim=claim.text), max_tokens=4096)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of limitations, got {type(raw).__name__}")
    texts = [str(item).strip() for item in raw if str(item).strip()]
    if not texts:
        logger.warning("LLM returned no atomic limitations for claim %d", claim.number)
        return []

    spans = _map_spans(claim.text, texts, llm)
    limitations = []
    for text in texts:
        span = spans.get(text)
        if span is None:
            span = _find_span(claim.text, text)
        limitations.append(AtomicLimitation(text=text, span=span))
    return limitations


def _map_spans(
    claim_text: str, limitations: list[str], llm: LLM
) -> dict[str, tuple[int, int]]:
    """Ask the LLM for limitation -> span mappings; keep only valid spans."""
    spans: dict[str, tuple[int, int]] = {}
    try:
        raw = llm.complete_json(
            MAP_LIMITATION_SPANS.format(
                claim_text=claim_text, limitations=json.dumps(limitations)
            ),
            max_tokens=4096,
        )
    except Exception:
        logger.warning("Span-mapping call failed; falling back to substring search", exc_info=True)
        return spans
    if not isinstance(raw, list):
        return spans
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("limitation", "")).strip()
        try:
            start, end = int(entry["start"]), int(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and 0 <= start < end <= len(claim_text):
            spans[text] = (start, end)
        else:
            logger.debug("Rejecting invalid span (%s, %s) for limitation %r", entry.get("start"), entry.get("end"), text)
    return spans


def _find_span(claim_text: str, limitation: str) -> Optional[tuple[int, int]]:
    """Locate ``limitation`` in ``claim_text`` case-insensitively."""
    index = claim_text.lower().find(limitation.lower())
    if index < 0:
        return None
    return (index, index + len(limitation))


def generate_keywords(
    patent: Patent,
    claims: Optional[list[Claim]] = None,
    llm: Optional[LLM] = None,
    votes: int = 3,
    top_n: int = 15,
) -> list[str]:
    """Generate search keywords from a patent's claims + description (LOW effort).

    Runs the keyword prompt ``votes`` times (at sampling temperature) and
    ranks the union by vote frequency, breaking ties by first appearance.
    Returns at most ``top_n`` lowercase keywords.
    """
    llm = llm or get_llm("low")
    claims = claims if claims is not None else patent.claims
    claims_text = "\n".join(c.text for c in claims)
    description = patent.abstract or (patent.specification or "")[:4000]
    prompt = KEYWORD_GENERATION.format(claims=claims_text, description=description)

    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for vote in range(max(1, votes)):
        try:
            raw = llm.complete_json(prompt, temperature=0.7, max_tokens=2048)
        except Exception:
            logger.warning("Keyword vote %d failed", vote, exc_info=True)
            continue
        if not isinstance(raw, list):
            continue
        seen_this_vote: set[str] = set()
        for item in raw:
            keyword = str(item).strip().lower()
            if not keyword or keyword in seen_this_vote:
                continue
            seen_this_vote.add(keyword)
            counts[keyword] += 1
            first_seen.setdefault(keyword, len(first_seen))
    ranked = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))
    return ranked[:top_n]
