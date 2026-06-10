"""Patent drafting skills: claim drafting, specification sections, and a
pure-python antecedent-basis check.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from patentkit.analysis.prompts import DRAFT_CLAIMS, DRAFT_SPEC_SECTION
from patentkit.llm.base import LLM, get_llm
from patentkit.models import Claim
from patentkit.parsing.claims import parse_claims

logger = logging.getLogger(__name__)

__all__ = ["draft_claims", "draft_spec_section", "check_antecedent_basis"]


def draft_claims(
    invention_disclosure: str,
    n_independent: int = 1,
    n_dependent: int = 5,
    llm: Optional[LLM] = None,
) -> list[Claim]:
    """Draft claims from an invention disclosure (HIGH effort).

    The LLM result is parsed through :func:`patentkit.parsing.claims.parse_claims`
    so callers get canonical :class:`~patentkit.models.Claim` objects with
    dependencies and element trees.
    """
    llm = llm or get_llm("high")
    response = llm.complete(
        DRAFT_CLAIMS.format(
            disclosure=invention_disclosure,
            n_independent=n_independent,
            n_dependent=n_dependent,
        ),
        max_tokens=8192,
    )
    claims = parse_claims(response.text)
    if not claims:
        logger.warning("Drafted claims could not be parsed; raw text:\n%s", response.text)
    return claims


def draft_spec_section(
    disclosure: str,
    section: str,
    claims: Optional[list[Claim]] = None,
    llm: Optional[LLM] = None,
) -> str:
    """Draft one specification section (e.g. "Summary", "Detailed Description")
    from a disclosure and optional drafted claims (HIGH effort)."""
    llm = llm or get_llm("high")
    claims_text = "\n\n".join(f"{c.number}. {c.text}" for c in claims) if claims else ""
    response = llm.complete(
        DRAFT_SPEC_SECTION.format(section=section, disclosure=disclosure, claims=claims_text),
        max_tokens=8192,
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Antecedent basis check (pure python)
# ---------------------------------------------------------------------------

#: tokens that terminate a noun phrase after a determiner (prepositions,
#: conjunctions, relative pronouns, common claim verbs/participles)
_PHRASE_BOUNDARY = frozenset(
    """
    a an the said of to for with in on at and or that which wherein whereby
    comprising comprises including includes having has being is are was were
    can may further such each when where while than then thereby thereof
    therein thereto configured adapted arranged operable coupled connected
    attached mounted disposed positioned formed defining defined extending
    extends received receiving based from by into onto over under between
    through via about within without
    """.split()
)

#: phrase heads that never need explicit antecedent introduction
_IGNORED_HEADS = frozenset({"same", "like", "following", "other", "invention", "claim", "claims"})

_TOKEN_RE = re.compile(r"[a-z0-9-]+|[^\sa-z0-9-]")
_MAX_PHRASE_TOKENS = 4


def _normalize(token: str) -> str:
    """Crude singular/plural normalization for head-noun matching."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _collect_phrase(tokens: list[str], start: int) -> list[str]:
    """Collect the noun phrase following a determiner at ``tokens[start]``."""
    phrase: list[str] = []
    for token in tokens[start + 1 :]:
        if not token[0].isalnum() and token != "-":
            break  # punctuation ends the phrase
        if token in _PHRASE_BOUNDARY:
            break
        phrase.append(token)
        if len(phrase) >= _MAX_PHRASE_TOKENS:
            break
    return phrase


def _scan_claim(text: str, introduced: set[str], issues: list[str], claim_number: int) -> None:
    """Scan one claim's tokens in order, updating ``introduced`` and ``issues``."""
    tokens = _TOKEN_RE.findall(text.lower())
    flagged: set[str] = set()
    for i, token in enumerate(tokens):
        if token in ("a", "an"):
            phrase = _collect_phrase(tokens, i)
            for word in phrase:
                if not word.endswith("ly"):  # skip adverbs like "removably"
                    introduced.add(_normalize(word))
        elif token in ("the", "said"):
            phrase = _collect_phrase(tokens, i)
            if not phrase:
                continue
            normalized = [_normalize(w) for w in phrase]
            if normalized[-1] in _IGNORED_HEADS:
                continue
            if not any(w in introduced for w in normalized):
                key = " ".join(phrase)
                if key not in flagged:
                    flagged.add(key)
                    issues.append(
                        f"Claim {claim_number}: '{token} {key}' may lack antecedent basis"
                    )


def check_antecedent_basis(claims: list[Claim]) -> list[str]:
    """Flag "the/said X" phrases lacking an "a/an X" antecedent (pure python).

    A simple lexical heuristic: noun phrases introduced with "a"/"an"
    contribute their (plural-normalized, non-adverb) tokens to the set of
    available antecedents; each "the X"/"said X" reference must share at
    least one token with that set. Dependent claims inherit antecedents from
    their full dependency chain, and references are checked in reading order
    (a "the X" *before* its "a X" is flagged).

    Known limitations (documented heuristic, not a grammar): no real noun-
    phrase parsing, so verbs inside phrases can leak into the antecedent set;
    one-token overlap matching accepts loose back-references ("the assembly"
    after "a gear assembly"); fixed expressions are handled via a small
    ignore list. Intended as a drafting aid, not a substitute for review.

    Returns a list of human-readable issue strings (empty when clean).
    """
    by_number = {c.number: c for c in claims}
    issues: list[str] = []
    for claim in claims:
        # Build the dependency chain root -> ... -> claim (cycle-safe).
        chain: list[Claim] = []
        current: Optional[Claim] = claim
        seen: set[int] = set()
        while current is not None and current.number not in seen:
            seen.add(current.number)
            chain.append(current)
            current = by_number.get(current.depends_on) if current.depends_on else None
        chain.reverse()

        introduced: set[str] = set()
        # Ancestors only contribute antecedents; issues are reported per claim.
        for ancestor in chain[:-1]:
            _scan_claim(ancestor.text, introduced, issues=[], claim_number=ancestor.number)
        _scan_claim(claim.text, introduced, issues, claim_number=claim.number)
    return issues
