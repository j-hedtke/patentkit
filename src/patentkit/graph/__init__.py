"""The gradual concept graph — the residue of successful searches.

No upfront ontology. Canonical concept nodes crystallize from agentic-search
behavior::

    non-graph agentic search -> search traces -> user feedback -> recurring
    successful matches -> candidate concept nodes -> reviewed/accepted edges
    -> better traversal

The core artifact is the :class:`MatchPair` ("query limitation X matched
patent passage Y"), harvested from the artifacts the system already emits
(agentic result rows, claim charts, traces, guided feedback). Accepted pairs
are clustered by phrase similarity and promoted through staged
:class:`ConceptNode` levels — never past stage 3 without a human
:func:`review`. The payoff is :func:`expand_limitation`: known aliases for a
limitation, injected into the agent's context as additional query angles.

Pure python + pydantic; file-backed stores (JSONL / JSON) behind thin
backend interfaces so ES/sqlite could replace them later.
"""

from patentkit.graph.expand import expand_limitation
from patentkit.graph.harvest import (
    harvest_from_claim_chart,
    harvest_from_reference_chart,
    harvest_from_results,
    harvest_from_trace,
    harvest_match_pairs,
)
from patentkit.graph.models import (
    STAGE_CANDIDATE_ALIAS_PAIR,
    STAGE_CANDIDATE_CLUSTER,
    STAGE_PRODUCTION,
    STAGE_RAW_PHRASE,
    STAGE_REVIEWED_CANONICAL,
    ConceptNode,
    MatchPair,
)
from patentkit.graph.promote import demote, promote, reject_alias, review
from patentkit.graph.similarity import normalize_phrase, phrase_similarity, token_jaccard
from patentkit.graph.store import (
    ConceptBackend,
    ConceptGraph,
    MatchPairBackend,
    MatchPairStore,
)

__all__ = [
    "MatchPair",
    "ConceptNode",
    "STAGE_RAW_PHRASE",
    "STAGE_CANDIDATE_ALIAS_PAIR",
    "STAGE_CANDIDATE_CLUSTER",
    "STAGE_REVIEWED_CANONICAL",
    "STAGE_PRODUCTION",
    "MatchPairBackend",
    "MatchPairStore",
    "ConceptBackend",
    "ConceptGraph",
    "harvest_match_pairs",
    "harvest_from_results",
    "harvest_from_claim_chart",
    "harvest_from_reference_chart",
    "harvest_from_trace",
    "promote",
    "review",
    "reject_alias",
    "demote",
    "expand_limitation",
    "normalize_phrase",
    "token_jaccard",
    "phrase_similarity",
]
