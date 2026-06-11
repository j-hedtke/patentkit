"""Pure-python phrase similarity for clustering and alias matching.

Normalized token Jaccard combined with :mod:`difflib` sequence ratio. An
embedding hook is supported everywhere via an optional ``embed`` callable
``(a, b) -> float`` — never a hard dependency.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: optional embedding-similarity hook: (phrase_a, phrase_b) -> similarity 0..1
EmbedFn = Callable[[str, str], float]

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_TOKEN_RE.split(str(text).lower())).strip()


def token_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the normalized token sets of ``a`` and ``b``."""
    tokens_a = set(normalize_phrase(a).split())
    tokens_b = set(normalize_phrase(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def phrase_similarity(a: str, b: str, *, embed: Optional[EmbedFn] = None) -> float:
    """Best of token Jaccard, difflib ratio, and the optional embedding hook."""
    norm_a, norm_b = normalize_phrase(a), normalize_phrase(b)
    if not norm_a or not norm_b:
        return 0.0
    score = max(token_jaccard(norm_a, norm_b),
                SequenceMatcher(None, norm_a, norm_b).ratio())
    if embed is not None:
        try:
            score = max(score, float(embed(a, b)))
        except Exception:  # noqa: BLE001 — the hook is best-effort
            logger.warning("embedding similarity hook failed; using lexical score",
                           exc_info=True)
    return score


__all__ = ["EmbedFn", "normalize_phrase", "token_jaccard", "phrase_similarity"]
