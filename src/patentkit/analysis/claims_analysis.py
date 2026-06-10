"""Claim-level analysis skills: interpretation, atomic-limitation splitting,
and search-keyword generation.

Every function accepts ``llm: LLM | None`` and defaults via
:func:`patentkit.llm.base.get_llm` at the documented effort tier.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from typing import Optional

from patentkit.analysis.prompts import (
    INTERPRET_CLAIM,
    KEYWORD_GENERATION,
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
    """Split ``claim`` into VERBATIM atomic limitations in document order.

    Each returned :class:`AtomicLimitation` is a contiguous, verbatim segment
    of ``claim.text`` with ``span`` set to its (start, end) character offsets,
    and the list is sorted by span start (preamble first, then each element in
    order). Verbatimness is ENFORCED in code, not just in the prompt: every
    LLM-returned limitation is located in the claim text after whitespace
    normalization (case-sensitively, then case-insensitively) and replaced by
    the exact claim slice. Items that cannot be located are snapped to the
    nearest deterministic structural segment, or dropped with a warning —
    paraphrased text is never kept.

    When ``llm`` is None, or when fewer than 2 returned limitations validate,
    a deterministic structural splitter is used instead (preamble up to the
    "comprising:"/"consisting of:" transition, then elements split on ";" /
    "; and", delimiters kept), so the function works keys-free.

    ``patent`` is currently unused beyond logging context but kept in the
    signature so callers can supply specification context later.
    """
    if llm is None:
        logger.info(
            "No LLM provided for claim %d; using deterministic structural splitter",
            claim.number,
        )
        return _structural_split(claim.text)

    raw = llm.complete_json(SPLIT_ATOMIC_LIMITATIONS.format(claim=claim.text), max_tokens=4096)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of limitations, got {type(raw).__name__}")
    texts = [str(item).strip() for item in raw if str(item).strip()]
    if not texts:
        logger.warning("LLM returned no atomic limitations for claim %d", claim.number)
        return []

    structural: Optional[list[AtomicLimitation]] = None
    validated: list[AtomicLimitation] = []
    seen_spans: set[tuple[int, int]] = set()
    for text in texts:
        span = _locate_verbatim(claim.text, text)
        if span is None:
            # Never keep paraphrased text: snap to the nearest deterministic
            # structural segment, or drop the item.
            if structural is None:
                structural = _structural_split(claim.text)
            snapped = _snap_to_segment(text, structural)
            if snapped is None:
                logger.warning(
                    "Dropping non-verbatim limitation %r (no close structural segment)", text
                )
                continue
            logger.warning("Limitation %r is not verbatim; snapped to %r", text, snapped.text)
            span = snapped.span
        assert span is not None
        if span in seen_spans:
            continue
        seen_spans.add(span)
        validated.append(AtomicLimitation(text=claim.text[span[0]:span[1]], span=span))

    if len(validated) < 2:
        logger.warning(
            "Only %d limitation(s) validated verbatim for claim %d; "
            "falling back to deterministic structural split",
            len(validated), claim.number,
        )
        return _structural_split(claim.text)

    validated.sort(key=lambda lim: lim.span[0])  # type: ignore[index]
    return validated


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
    text: str, segments: list[AtomicLimitation]
) -> Optional[AtomicLimitation]:
    """Snap a non-verbatim limitation to its closest structural segment, or
    None when nothing is close enough (similarity ratio below 0.6)."""
    norm = " ".join(text.split()).lower()
    best: Optional[AtomicLimitation] = None
    best_ratio = 0.0
    for segment in segments:
        seg_norm = " ".join(segment.text.split()).lower()
        ratio = difflib.SequenceMatcher(None, norm, seg_norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = segment, ratio
    if best is not None and best_ratio >= 0.6:
        return best
    return None


#: Transitional phrase ending a US claim preamble; the delimiter stays with
#: the preamble ("... comprising:" / "... consisting of:").
_PREAMBLE_RE = re.compile(r"\b(?:compris(?:ing|es)|consist(?:ing|s)\s+of)\s*:?", re.IGNORECASE)

#: Element delimiter: ";" optionally followed by the conjunction "and", which
#: stays with the preceding element ("...; and").
_ELEMENT_DELIM_RE = re.compile(r";(?:\s+and\b)?")


def _structural_split(claim_text: str) -> list[AtomicLimitation]:
    """Deterministically split a US-style claim into verbatim segments.

    The preamble ends at the transitional phrase ("comprising:" etc., kept
    with the preamble); elements are then split on ";" / "; and" (delimiters
    kept with the preceding element). Whitespace is trimmed from each segment
    but spans always index into the original claim text.
    """
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    preamble = _PREAMBLE_RE.search(claim_text)
    if preamble:
        boundaries.append((0, preamble.end()))
        cursor = preamble.end()
    for match in _ELEMENT_DELIM_RE.finditer(claim_text, cursor):
        boundaries.append((cursor, match.end()))
        cursor = match.end()
    if claim_text[cursor:].strip():
        boundaries.append((cursor, len(claim_text)))

    limitations: list[AtomicLimitation] = []
    for start, end in boundaries:
        segment = claim_text[start:end]
        s = start + (len(segment) - len(segment.lstrip()))
        e = end - (len(segment) - len(segment.rstrip()))
        if e > s:
            limitations.append(AtomicLimitation(text=claim_text[s:e], span=(s, e)))
    return limitations


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
