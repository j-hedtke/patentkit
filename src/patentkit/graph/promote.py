"""Staged promotion: accepted match pairs crystallize into concept nodes.

``promote`` clusters accepted match pairs by limitation-phrase similarity
(pure python — normalized token Jaccard + difflib ratio; an embedding hook
is an optional callable, never a hard dependency), then creates/advances
:class:`~patentkit.graph.models.ConceptNode` objects with evidence counts.

Stage gates (cumulative):

- stage 1 (candidate alias pair): >= 2 distinct surface phrasings, or >= 2
  accepted pairs;
- stage 2 (candidate cluster): >= ``min_accepted`` accepted pairs across
  >= 2 searches;
- stage 3 (canonical): >= ``min_searches`` searches, >= ``min_accepted``
  accepted pairs, >= ``min_patents`` distinct patents, reject rate <=
  ``max_reject_rate``.

Promotion NEVER advances past stage 3. Stage 4 (production traversal)
requires an explicit human :func:`review` with ``approved=True``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from patentkit.graph.models import (
    STAGE_CANDIDATE_ALIAS_PAIR,
    STAGE_CANDIDATE_CLUSTER,
    STAGE_PRODUCTION,
    STAGE_RAW_PHRASE,
    STAGE_REVIEWED_CANONICAL,
    ConceptNode,
    MatchPair,
)
from patentkit.graph.similarity import EmbedFn, normalize_phrase, phrase_similarity
from patentkit.graph.store import ConceptBackend, MatchPairBackend

#: default phrase-similarity threshold for joining a cluster / matching a node
DEFAULT_SIMILARITY = 0.6

_NAME_RE = re.compile(r"[^A-Z0-9]+")


def _canonical_name(phrase: str, max_length: int = 60) -> str:
    """``"rank documents by embedding similarity"`` ->
    ``"RANK_DOCUMENTS_BY_EMBEDDING_SIMILARITY"``."""
    name = _NAME_RE.sub("_", phrase.upper()).strip("_")
    return name[:max_length].rstrip("_") or "CONCEPT"


class _Cluster:
    """Accepted pairs whose limitation phrasings express one concept."""

    def __init__(self) -> None:
        self.pairs: list[MatchPair] = []
        self.phrases: list[str] = []  # distinct surface phrasings, first-seen order

    def add(self, pair: MatchPair) -> None:
        self.pairs.append(pair)
        if normalize_phrase(pair.query_limitation) not in (
                normalize_phrase(p) for p in self.phrases):
            self.phrases.append(pair.query_limitation.strip())

    def similarity(self, phrase: str, embed: Optional[EmbedFn]) -> float:
        return max(phrase_similarity(phrase, existing, embed=embed)
                   for existing in self.phrases)


def _cluster_accepted(accepted: list[MatchPair], threshold: float,
                      embed: Optional[EmbedFn]) -> list[_Cluster]:
    """Greedy agglomerative clustering by limitation-phrase similarity."""
    clusters: list[_Cluster] = []
    for pair in accepted:
        best, best_score = None, 0.0
        for cluster in clusters:
            score = cluster.similarity(pair.query_limitation, embed)
            if score > best_score:
                best, best_score = cluster, score
        if best is not None and best_score >= threshold:
            best.add(pair)
        else:
            cluster = _Cluster()
            cluster.add(pair)
            clusters.append(cluster)
    return clusters


def _target_stage(*, aliases: int, accepted: int, searches: int, patents: int,
                  rejected: int, min_searches: int, min_accepted: int,
                  max_reject_rate: float, min_patents: int) -> int:
    stage = STAGE_RAW_PHRASE
    if aliases >= 2 or accepted >= 2:
        stage = STAGE_CANDIDATE_ALIAS_PAIR
    if accepted >= min_accepted and searches >= 2:
        stage = STAGE_CANDIDATE_CLUSTER
    reject_rate = rejected / max(accepted + rejected, 1)
    if (searches >= min_searches and accepted >= min_accepted
            and patents >= min_patents and reject_rate <= max_reject_rate):
        stage = STAGE_REVIEWED_CANONICAL
    return stage  # never STAGE_PRODUCTION — that requires review()


def promote(graph: ConceptBackend, store: MatchPairBackend, *,
            min_searches: int = 5, min_accepted: int = 3,
            max_reject_rate: float = 0.2, min_patents: int = 3,
            similarity_threshold: float = DEFAULT_SIMILARITY,
            embed: Optional[EmbedFn] = None) -> list[ConceptNode]:
    """Cluster accepted match pairs and create/advance concept nodes.

    Evidence is recomputed from the full store each call (idempotent);
    stages only move up (existing reviewed stages are never lowered here —
    use :func:`demote` / :func:`reject_alias` for that), and auto-promotion
    is capped at stage 3. The graph is saved before returning.

    Args:
        graph: the concept graph to update.
        store: the match-pair store to read evidence from.
        min_searches: distinct searches required for stage 3.
        min_accepted: accepted pairs required for stages 2 and 3.
        max_reject_rate: maximum rejected/(accepted+rejected) for stage 3.
        min_patents: distinct patents required for stage 3.
        similarity_threshold: phrase-similarity threshold for clustering and
            for attributing rejected pairs to a cluster.
        embed: optional ``(a, b) -> similarity`` embedding hook.

    Returns:
        The concept nodes created or updated, in cluster order.
    """
    pairs = list(store.iter())
    accepted = [p for p in pairs
                if p.outcome == "accepted" and normalize_phrase(p.query_limitation)]
    rejected = [p for p in pairs
                if p.outcome == "rejected" and normalize_phrase(p.query_limitation)]

    touched: list[ConceptNode] = []
    for cluster in _cluster_accepted(accepted, similarity_threshold, embed):
        searches = {p.search_id for p in cluster.pairs if p.search_id} or \
            {str(i) for i in range(len(cluster.pairs))}
        patents = sorted({p.patent_id for p in cluster.pairs if p.patent_id})
        rejected_n = sum(
            1 for p in rejected
            if cluster.similarity(p.query_limitation, embed) >= similarity_threshold)

        # representative phrase: most frequent phrasing among accepted pairs
        counts = Counter(normalize_phrase(p.query_limitation) for p in cluster.pairs)
        representative = max(cluster.phrases,
                             key=lambda phrase: counts[normalize_phrase(phrase)])

        node = None
        for phrase in cluster.phrases:
            node = graph.find_by_alias(phrase)
            if node is not None:
                break
        if node is None:
            node = graph.add(ConceptNode(canonical_name=_canonical_name(representative)))

        known = {normalize_phrase(a) for a in node.aliases}
        rejected_aliases = {normalize_phrase(a)
                            for a in node.evidence.get("rejected_aliases", [])}
        for phrase in cluster.phrases:
            norm = normalize_phrase(phrase)
            if norm not in known and norm not in rejected_aliases:
                node.aliases.append(phrase)
                known.add(norm)

        node.evidence.update({
            "searches": len(searches),
            "accepted_charts": len(cluster.pairs),
            "rejected": rejected_n,
            "patents": patents,
        })
        target = _target_stage(
            aliases=len(node.aliases), accepted=len(cluster.pairs),
            searches=len(searches), patents=len(patents), rejected=rejected_n,
            min_searches=min_searches, min_accepted=min_accepted,
            max_reject_rate=max_reject_rate, min_patents=min_patents)
        if node.stage != STAGE_PRODUCTION:  # human-reviewed nodes stay put
            node.stage = max(node.stage, min(target, STAGE_REVIEWED_CANONICAL))
        touched.append(node)

    graph.save()
    return touched


# ------------------------------------------------------- human in the loop

def review(node: ConceptNode, approved: bool = True) -> ConceptNode:
    """Human review gate. ``approved=True`` promotes a stage-3 node to stage
    4 (production traversal) — the ONLY path to stage 4. ``approved=False``
    demotes the node back to a candidate cluster (stage <= 2)."""
    if approved:
        if node.stage < STAGE_REVIEWED_CANONICAL:
            raise ValueError(
                f"{node.canonical_name} is at stage {node.stage}; only stage "
                f"{STAGE_REVIEWED_CANONICAL} nodes can be approved for production")
        node.stage = STAGE_PRODUCTION
    else:
        node.stage = min(node.stage, STAGE_CANDIDATE_CLUSTER)
    return node


def reject_alias(node: ConceptNode, alias: str) -> ConceptNode:
    """Remove a bad synonym from a node and remember it so promotion never
    re-adds it. Drops the node to stage 1 if fewer than two aliases remain."""
    norm = normalize_phrase(alias)
    kept = [a for a in node.aliases if normalize_phrase(a) != norm]
    if len(kept) == len(node.aliases):
        return node
    node.aliases = kept
    node.evidence.setdefault("rejected_aliases", []).append(alias)
    if node.stage >= STAGE_CANDIDATE_CLUSTER and len(kept) < 2:
        node.stage = STAGE_CANDIDATE_ALIAS_PAIR
    return node


def demote(node: ConceptNode, stage: int = STAGE_CANDIDATE_CLUSTER) -> ConceptNode:
    """Lower a node's stage (never raises it)."""
    node.stage = max(STAGE_RAW_PHRASE, min(node.stage, int(stage)))
    return node


__all__ = ["promote", "review", "reject_alias", "demote", "DEFAULT_SIMILARITY"]
