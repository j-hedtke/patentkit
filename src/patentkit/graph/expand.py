"""The traversal payoff: expand a limitation into known concept aliases.

``expand_limitation`` looks a limitation up against the concept graph
(exact normalized alias hit, alias-contained-in-limitation, or fuzzy phrase
match) and returns the matching concepts' other aliases — ready to inject
into an agent's context as additional query angles.
"""

from __future__ import annotations

from typing import Any, Optional

from patentkit.graph.models import STAGE_REVIEWED_CANONICAL, ConceptNode
from patentkit.graph.promote import DEFAULT_SIMILARITY
from patentkit.graph.similarity import EmbedFn, normalize_phrase, phrase_similarity
from patentkit.graph.store import ConceptBackend


def _node_matches(node: ConceptNode, text: str, norm: str, threshold: float,
                  embed: Optional[EmbedFn]) -> bool:
    for alias in [node.canonical_name.replace("_", " "), *node.aliases]:
        alias_norm = normalize_phrase(alias)
        if not alias_norm:
            continue
        if alias_norm == norm:
            return True
        # a multi-word alias appearing inside a longer limitation counts
        if len(alias_norm.split()) >= 2 and alias_norm in norm:
            return True
        if phrase_similarity(alias, text, embed=embed) >= threshold:
            return True
    return False


def expand_limitation(graph: ConceptBackend, limitation_text: Any, *,
                      min_stage: int = STAGE_REVIEWED_CANONICAL,
                      similarity_threshold: float = DEFAULT_SIMILARITY,
                      embed: Optional[EmbedFn] = None) -> list[str]:
    """Known aliases for the concepts a limitation expresses.

    Args:
        graph: the concept graph.
        limitation_text: a limitation as a plain string (objects exposing
            ``.text`` are accepted too).
        min_stage: only concepts at this stage or above contribute (default:
            stage 3, reviewed canonical).
        similarity_threshold: fuzzy-match threshold against aliases.
        embed: optional ``(a, b) -> similarity`` embedding hook.

    Returns:
        Alias phrases (deduplicated, graph order), excluding phrasings that
        normalize to the input itself.
    """
    text = getattr(limitation_text, "text", None)
    text = str(text) if text is not None else str(limitation_text)
    norm = normalize_phrase(text)
    if not norm:
        return []
    expansions: list[str] = []
    seen: set[str] = {norm}
    for node in graph.nodes():
        if node.stage < min_stage:
            continue
        if not _node_matches(node, text, norm, similarity_threshold, embed):
            continue
        for alias in node.aliases:
            alias_norm = normalize_phrase(alias)
            if alias_norm and alias_norm not in seen:
                expansions.append(alias)
                seen.add(alias_norm)
    return expansions


__all__ = ["expand_limitation"]
