"""Pure-python parsing of patent claim text into the canonical claim model.

No LLMs, no optional dependencies: numbered-claim splitting, dependency
detection, deterministic limitation splitting (the PRIMARY mechanism for
producing :class:`~patentkit.models.Limitation` units at parse time), and a
heuristic element tree built from claim punctuation conventions (":"
preamble, ";" elements, "wherein"/"whereby" clauses).
"""

from __future__ import annotations

import logging
import re

from patentkit.models import Claim, ClaimElement, Limitation

logger = logging.getLogger(__name__)

#: start of a numbered claim: "1." or "1 ." at the beginning of a line
_CLAIM_START_RE = re.compile(r"(?m)^\s*(\d{1,3})\s*\.\s+")

#: dependency phrases: "claim 3", "claims 1-4", "any of claims 2 to 5", "claim 1 or 2"
_DEP_RE = re.compile(
    r"\bclaims?\s+(\d+)(?:\s*(?:-|–|to|through|or)\s*(\d+))?",
    re.IGNORECASE,
)

#: split point immediately before a "wherein"/"whereby" clause
_WHEREIN_RE = re.compile(r",?\s+(?=\bwhere(?:in|by)\b)", re.IGNORECASE)

#: anchors that introduce the claims section of a specification
_CLAIMS_ANCHORS = [
    re.compile(r"what\s+is\s+claimed\s+is\s*:?", re.IGNORECASE),
    re.compile(r"the\s+invention\s+claimed\s+is\s*:?", re.IGNORECASE),
    re.compile(r"\bI\s+claim\s*:?", re.IGNORECASE),
    re.compile(r"\bwe\s+claim\s*:?", re.IGNORECASE),
    re.compile(r"(?m)^\s*CLAIMS?\s*$"),
]


def parse_claims(claims_text: str) -> list[Claim]:
    """Parse a block of claim text into :class:`~patentkit.models.Claim` objects.

    Claims are split on numbered anchors at line starts ("1." / "1 ." styles).
    Spurious mid-text numbers are rejected by requiring claim numbers to be
    strictly increasing. Dependency is detected from "claim N" / "any of
    claims N-M" phrases (the first referenced claim number wins). Each claim
    gets a heuristic :class:`~patentkit.models.ClaimElement` tree.
    """
    matches = list(_CLAIM_START_RE.finditer(claims_text))
    # Keep only strictly-increasing claim numbers to drop spurious matches
    # (e.g. an enumerated list inside a claim body).
    accepted: list[re.Match[str]] = []
    last_number = 0
    for match in matches:
        number = int(match.group(1))
        if number > last_number:
            accepted.append(match)
            last_number = number

    claims: list[Claim] = []
    for i, match in enumerate(accepted):
        start = match.end()
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(claims_text)
        number = int(match.group(1))
        text = claims_text[start:end].strip()
        if not text:
            continue
        depends_on = _detect_dependency(text)
        if depends_on is not None and depends_on >= number:
            logger.debug("Claim %d references claim %d (not a dependency); ignoring", number, depends_on)
            depends_on = None
        claims.append(
            Claim(
                number=number,
                text=text,
                depends_on=depends_on,
                elements=_build_element_tree(text),
                limitations=split_limitations(text, number),
            )
        )
    return claims


#: Transitional phrase ending a US claim preamble; the delimiter stays with
#: the preamble ("... comprising:" / "... consisting of:").
_PREAMBLE_RE = re.compile(r"\b(?:compris(?:ing|es)|consist(?:ing|s)\s+of)\s*:?", re.IGNORECASE)

#: Element delimiter: ";" optionally followed by the conjunction "and", which
#: stays with the preceding element ("...; and").
_ELEMENT_DELIM_RE = re.compile(r";(?:\s+and\b)?")


def element_label(claim_number: int, index: int) -> str:
    """Bracket-letter label for the ``index``-th (0-based) claim element:
    "1[a]", "1[b]", ... "1[z]", "1[aa]", ..."""
    letters = ""
    i = index + 1
    while i:
        i, remainder = divmod(i - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"{claim_number}[{letters}]"


def split_limitations(claim_text: str, claim_number: int) -> list[Limitation]:
    """Deterministically split a US-style claim into verbatim limitation units.

    This is the PRIMARY limitation-splitting mechanism, run at index/parse
    time (no LLM): the preamble ends at the transitional phrase
    ("comprising:" / "consisting of:", kept with the preamble) and gets the
    label "N[pre]"; the remaining elements are split on ";" / "; and"
    (delimiters kept with the preceding element) and labeled "N[a]", "N[b]",
    ... in document order. A claim with no transitional phrase and no
    semicolons yields a single limitation. Whitespace is trimmed from each
    segment, but every ``text`` is a verbatim slice of ``claim_text`` and
    every ``span`` indexes into the original claim text.
    """
    boundaries: list[tuple[int, int, bool]] = []  # (start, end, is_preamble)
    cursor = 0
    preamble = _PREAMBLE_RE.search(claim_text)
    if preamble:
        boundaries.append((0, preamble.end(), True))
        cursor = preamble.end()
    for match in _ELEMENT_DELIM_RE.finditer(claim_text, cursor):
        boundaries.append((cursor, match.end(), False))
        cursor = match.end()
    if claim_text[cursor:].strip():
        boundaries.append((cursor, len(claim_text), False))

    limitations: list[Limitation] = []
    element_index = 0
    for start, end, is_preamble in boundaries:
        segment = claim_text[start:end]
        s = start + (len(segment) - len(segment.lstrip()))
        e = end - (len(segment) - len(segment.rstrip()))
        if e <= s:
            continue
        if is_preamble:
            label = f"{claim_number}[pre]"
        else:
            label = element_label(claim_number, element_index)
            element_index += 1
        limitations.append(Limitation(label=label, text=claim_text[s:e], span=(s, e)))
    return limitations


def _detect_dependency(claim_text: str) -> int | None:
    """Return the first claim number referenced by a dependency phrase, if any."""
    match = _DEP_RE.search(claim_text)
    return int(match.group(1)) if match else None


def _build_element_tree(claim_text: str) -> list[ClaimElement]:
    """Build a heuristic element tree from claim punctuation.

    Split on the first ":" (preamble vs. body), then ";" (elements), then
    "wherein"/"whereby" clauses (children of their element). When a preamble
    is present, it becomes the single root node and the body elements are
    its children.
    """
    text = " ".join(claim_text.split())
    if ":" in text:
        preamble, _, body = text.partition(":")
        children = [_build_element(chunk) for chunk in _split_semicolons(body)]
        return [ClaimElement(text=preamble.strip() + ":", children=children)]
    return [_build_element(chunk) for chunk in _split_semicolons(text)]


def _split_semicolons(body: str) -> list[str]:
    chunks = []
    for raw in body.split(";"):
        chunk = raw.strip().strip(",").strip()
        chunk = re.sub(r"^(?:and|or)\s+", "", chunk, flags=re.IGNORECASE)
        if chunk:
            chunks.append(chunk)
    return chunks


def _build_element(chunk: str) -> ClaimElement:
    """Split trailing wherein/whereby clauses out of ``chunk`` as children."""
    parts = [p.strip() for p in _WHEREIN_RE.split(chunk) if p.strip()]
    if not parts:
        return ClaimElement(text=chunk.strip())
    main, rest = parts[0], parts[1:]
    children = [ClaimElement(text=p) for p in rest]
    return ClaimElement(text=main, children=children)


def extract_claims_section(spec_text: str) -> str:
    """Return the claims portion of a full specification text.

    Looks for the conventional anchors "What is claimed is:", "I claim",
    "We claim", "The invention claimed is", or a standalone "CLAIMS" heading,
    and returns everything after the *last* anchor occurrence. If no anchor is
    found, the input is returned unchanged (caller may still run
    :func:`parse_claims` over it).
    """
    best_end = -1
    for anchor in _CLAIMS_ANCHORS:
        for match in anchor.finditer(spec_text):
            best_end = max(best_end, match.end())
    if best_end < 0:
        logger.debug("No claims-section anchor found; returning full text")
        return spec_text
    return spec_text[best_end:].strip()


def claim_element_outline(claim: Claim) -> str:
    """Render a claim's element tree as an indented plain-text outline."""
    lines = [f"Claim {claim.number}:"]

    def walk(element: ClaimElement, depth: int) -> None:
        lines.append(f"{'  ' * depth}- {element.text}")
        for child in element.children:
            walk(child, depth + 1)

    if claim.elements:
        for element in claim.elements:
            walk(element, 1)
    else:
        lines.append(f"  - {claim.text}")
    return "\n".join(lines)
