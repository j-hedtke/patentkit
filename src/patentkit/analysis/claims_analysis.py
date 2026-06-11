"""Claim-level analysis skills: interpretation, optional limitation
refinement, and search-keyword generation.

Limitation SPLITTING is not an analysis step: claims carry precomputed
:class:`~patentkit.models.Limitation` units assigned at parse time by the
deterministic splitter in :func:`patentkit.parsing.claims.split_limitations`
(reachable via :meth:`~patentkit.models.Claim.get_limitations`).
:func:`refine_limitations` is an OPTIONAL LLM pass that may merge/split
those deterministic segments, with verbatimness enforced in code.

Every LLM-backed function accepts ``llm: LLM | None`` and defaults via
:func:`patentkit.llm.base.get_llm` at the documented effort tier.
"""

from __future__ import annotations

import difflib
import logging
from collections import Counter
from typing import Optional

from patentkit.analysis.prompts import (
    INTERPRET_CLAIM,
    KEYWORD_GENERATION,
    REFINE_LIMITATIONS,
)
from patentkit.llm.base import LLM, get_llm
from patentkit.models import Claim, Limitation, Patent
from patentkit.parsing.claims import element_label

logger = logging.getLogger(__name__)

__all__ = ["interpret_claim", "refine_limitations", "generate_keywords"]


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


def refine_limitations(claim: Claim, llm: Optional[LLM] = None) -> list[Limitation]:
    """OPTIONALLY refine ``claim``'s precomputed limitations with an LLM.

    The deterministic structural units from
    :meth:`~patentkit.models.Claim.get_limitations` are the baseline; the LLM
    may merge or split them (e.g. join an element that genuinely spans a
    semicolon), but the result must remain VERBATIM, in-order, contiguous
    segments of ``claim.text``. Verbatimness is ENFORCED in code, not just in
    the prompt: every LLM-returned segment is located in the claim text after
    whitespace normalization (case-sensitively, then case-insensitively) and
    replaced by the exact claim slice. Segments that cannot be located are
    snapped to the nearest deterministic structural unit, or dropped with a
    warning — paraphrased text is never kept.

    When ``llm`` is None, or when fewer than 2 returned segments validate,
    the deterministic limitations are returned unchanged — chart building
    never REQUIRES an LLM for splitting.

    Labels are reassigned in document order: a refined segment that exactly
    matches the deterministic preamble keeps "N[pre]"; all other segments get
    "N[a]", "N[b]", ...
    """
    base = claim.get_limitations()
    if llm is None:
        return base

    segments = "\n".join(f"- {lim.text}" for lim in base)
    raw = llm.complete_json(
        REFINE_LIMITATIONS.format(claim=claim.text, segments=segments), max_tokens=4096
    )
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of limitations, got {type(raw).__name__}")
    texts = [str(item).strip() for item in raw if str(item).strip()]
    if not texts:
        logger.warning("LLM returned no limitations for claim %d; keeping the "
                       "deterministic split", claim.number)
        return base

    validated_spans: list[tuple[int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for text in texts:
        span = _locate_verbatim(claim.text, text)
        if span is None:
            # Never keep paraphrased text: snap to the nearest deterministic
            # structural unit, or drop the item.
            snapped = _snap_to_segment(text, base)
            if snapped is None or snapped.span is None:
                logger.warning(
                    "Dropping non-verbatim limitation %r (no close structural segment)", text
                )
                continue
            logger.warning("Limitation %r is not verbatim; snapped to %r", text, snapped.text)
            span = snapped.span
        if span in seen_spans:
            continue
        seen_spans.add(span)
        validated_spans.append(span)

    if len(validated_spans) < 2:
        logger.warning(
            "Only %d limitation(s) validated verbatim for claim %d; "
            "keeping the deterministic structural split",
            len(validated_spans), claim.number,
        )
        return base

    validated_spans.sort(key=lambda span: span[0])
    preamble_span = (
        base[0].span if base and base[0].label.endswith("[pre]") else None
    )
    refined: list[Limitation] = []
    element_index = 0
    for span in validated_spans:
        if preamble_span is not None and span == preamble_span:
            label = f"{claim.number}[pre]"
        else:
            label = element_label(claim.number, element_index)
            element_index += 1
        refined.append(Limitation(label=label, text=claim.text[span[0]:span[1]], span=span))
    return refined


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, keeping an index map back to
    the original text (normalized index -> original index)."""
    chars: list[str] = []
    index_map: list[int] = []
    pending_space = -1  # original index of a pending collapsed space
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars:  # drop leading whitespace entirely
                pending_space = i if pending_space < 0 else pending_space
            continue
        if pending_space >= 0:
            chars.append(" ")
            index_map.append(pending_space)
            pending_space = -1
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def _locate_verbatim(claim_text: str, limitation: str) -> Optional[tuple[int, int]]:
    """Locate ``limitation`` as a contiguous segment of ``claim_text`` after
    whitespace normalization; case-sensitive first, then case-insensitive.

    Returns (start, end) offsets into the ORIGINAL claim text, or None.
    """
    needle = " ".join(limitation.split())
    if not needle:
        return None
    haystack, index_map = _normalize_with_map(claim_text)
    pos = haystack.find(needle)
    if pos < 0:
        pos = haystack.lower().find(needle.lower())
    if pos < 0:
        return None
    start = index_map[pos]
    end = index_map[pos + len(needle) - 1] + 1
    return (start, end)


def _snap_to_segment(
    text: str, segments: list[Limitation]
) -> Optional[Limitation]:
    """Snap a non-verbatim limitation to its closest structural segment, or
    None when nothing is close enough (similarity ratio below 0.6)."""
    norm = " ".join(text.split()).lower()
    best: Optional[Limitation] = None
    best_ratio = 0.0
    for segment in segments:
        seg_norm = " ".join(segment.text.split()).lower()
        ratio = difflib.SequenceMatcher(None, norm, seg_norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = segment, ratio
    if best is not None and best_ratio >= 0.6:
        return best
    return None


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
