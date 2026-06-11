"""Core graph artifacts: match pairs and staged concept nodes.

Deliberately dependency-light (pydantic only) and decoupled from the
``patentkit.models`` data model: limitation text arrives as plain strings
(harvesters duck-type ``.text`` off richer limitation objects).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

#: stage 0 — a raw surface phrase observed in matches
STAGE_RAW_PHRASE = 0
#: stage 1 — two surface phrasings linked as a candidate alias pair
STAGE_CANDIDATE_ALIAS_PAIR = 1
#: stage 2 — a candidate cluster with repeated accepted evidence
STAGE_CANDIDATE_CLUSTER = 2
#: stage 3 — canonical node: all evidence gates passed (auto-promotion cap)
STAGE_REVIEWED_CANONICAL = 3
#: stage 4 — production traversal node; requires explicit human review
STAGE_PRODUCTION = 4

STAGE_LABELS = {
    STAGE_RAW_PHRASE: "raw phrase",
    STAGE_CANDIDATE_ALIAS_PAIR: "candidate alias pair",
    STAGE_CANDIDATE_CLUSTER: "candidate cluster",
    STAGE_REVIEWED_CANONICAL: "reviewed canonical",
    STAGE_PRODUCTION: "production traversal node",
}


class MatchPair(BaseModel):
    """One observed match: "query limitation X matched patent passage Y".

    The core artifact the graph crystallizes from. Harvested with
    ``outcome="unreviewed"``; user feedback flips pairs to ``accepted`` /
    ``rejected`` (see :meth:`~patentkit.graph.store.MatchPairStore.mark_outcome`).
    """

    #: text of the query patent's limitation (plain string)
    query_limitation: str
    #: the passage/quote from the matched patent
    matched_text: str
    patent_id: str
    #: e.g. "claim 1", "specification"
    section: str = ""
    #: accepted | rejected | unreviewed
    outcome: str = "unreviewed"
    #: teaches_limitation | wrong_field | right_field_wrong_mechanism |
    #: cumulative | bad_synonym | ...
    feedback_type: str = ""
    search_id: str = ""
    embedding_similarity: Optional[float] = None
    bm25_score: Optional[float] = None
    #: ISO date string, caller-supplied
    created_at: str = ""


class ConceptNode(BaseModel):
    """A canonical concept that crystallized from recurring matches.

    Stages (see the ``STAGE_*`` constants): 0 raw phrase, 1 candidate alias
    pair, 2 candidate cluster, 3 reviewed canonical, 4 production traversal
    node. :func:`patentkit.graph.promote.promote` never advances past stage
    3; stage 4 requires :func:`patentkit.graph.promote.review`.
    """

    #: e.g. "RANK_DOCUMENTS_BY_EMBEDDING_SIMILARITY"
    canonical_name: str
    #: surface phrases that express this concept
    aliases: list[str] = Field(default_factory=list)
    stage: int = STAGE_RAW_PHRASE
    #: {"searches": int, "accepted_charts": int, "rejected": int, "patents": [..]}
    evidence: dict = Field(default_factory=lambda: {
        "searches": 0, "accepted_charts": 0, "rejected": 0, "patents": [],
    })


__all__ = [
    "MatchPair",
    "ConceptNode",
    "STAGE_RAW_PHRASE",
    "STAGE_CANDIDATE_ALIAS_PAIR",
    "STAGE_CANDIDATE_CLUSTER",
    "STAGE_REVIEWED_CANONICAL",
    "STAGE_PRODUCTION",
    "STAGE_LABELS",
]
