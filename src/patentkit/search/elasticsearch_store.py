"""Elasticsearch backend implementing both KeywordStore and VectorStore.

One index holds the canonical :class:`~patentkit.models.Patent` documents
(serialized via ``model_dump(mode="json")``) plus flattened search fields
(``claims_text``, ``cpc``, ``inventor_names``, ...) and, when an embedding
provider is configured, nested specification chunks with ``dense_vector``
embeddings for kNN retrieval.

Query construction lives in pure functions (:func:`build_es_query`,
:func:`build_es_filters`, :func:`build_knn_query`, :func:`patent_to_doc`,
:func:`build_es_mapping`) so the full ``SearchQuery`` translation is unit
testable without the ``elasticsearch`` package or a running cluster.

The ``elasticsearch`` client library is an optional extra::

    pip install 'patentkit[elasticsearch]'
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from patentkit.config import resolve_key
from patentkit.models import Patent, PatentNumber
from patentkit.search.base import EmbeddingProvider, Passage, SearchQuery, SearchResult
from patentkit.search.vector import chunk_text

logger = logging.getLogger(__name__)

#: per-field boosts, mirroring the in-memory BM25 store
FIELD_BOOSTS: dict[str, float] = {
    "title": 2.0,
    "abstract": 1.2,
    "claims": 1.5,
    "specification": 1.0,
}

#: SearchQuery field name -> Elasticsearch field name
_ES_FIELD_NAMES: dict[str, str] = {
    "title": "title",
    "abstract": "abstract",
    "claims": "claims_text",
    "specification": "specification",
}
_ES_FIELD_NAMES_REVERSE = {v: k for k, v in _ES_FIELD_NAMES.items()}

PHRASE_BOOST = 2.0
FUZZINESS = "AUTO"
PHRASE_SLOP = 1


def _import_elasticsearch():
    """Lazily import the optional ``elasticsearch`` package."""
    try:
        import elasticsearch  # noqa: F401
        return elasticsearch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The Elasticsearch backend requires the 'elasticsearch' package. "
            "Install it with: pip install 'patentkit[elasticsearch]'"
        ) from exc


def normalized_number(number: PatentNumber) -> str:
    """Kind-code-insensitive document id, e.g. ``US7654321`` for ``US7654321B2``."""
    return f"{number.country_code}{number.number.lstrip('0') or '0'}"


def es_search_fields(fields: list[str]) -> list[str]:
    """Translate SearchQuery field names to boosted ES field expressions."""
    out = []
    for name in fields:
        es_name = _ES_FIELD_NAMES.get(name, name)
        boost = FIELD_BOOSTS.get(name, 1.0)
        out.append(f"{es_name}^{boost}")
    return out


def _keyword_clauses(keyword: str, search_fields: list[str]) -> list[dict]:
    """best_fields match (fuzzy) + phrase match with boost, per production strategy."""
    return [
        {
            "multi_match": {
                "query": keyword,
                "fields": search_fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": FUZZINESS,
            }
        },
        {
            "multi_match": {
                "query": keyword,
                "fields": search_fields,
                "type": "phrase",
                "boost": PHRASE_BOOST,
                "slop": PHRASE_SLOP,
            }
        },
    ]


def build_es_filters(query: SearchQuery) -> list[dict]:
    """Translate SearchQuery metadata constraints into ES filter clauses.

    Covers: countries, include_numbers, art class prefixes, inventors,
    assignees, and before/after date cutoffs (priority_date falling back to
    filing_date). Exclusions live in ``must_not`` — see :func:`build_es_query`.
    """
    filters: list[dict] = []
    if query.countries:
        filters.append({"terms": {"country": query.countries}})
    if query.include_numbers:
        filters.append(
            {"terms": {"patent_number_norm": [normalized_number(n) for n in query.include_numbers]}}
        )
    if query.art_classes:
        filters.append(
            {
                "bool": {
                    "should": [{"prefix": {"cpc": prefix}} for prefix in query.art_classes],
                    "minimum_should_match": 1,
                }
            }
        )
    if query.inventors:
        filters.append(
            {
                "bool": {
                    "should": [{"match": {"inventor_names": name}} for name in query.inventors],
                    "minimum_should_match": 1,
                }
            }
        )
    if query.assignees:
        filters.append(
            {
                "bool": {
                    "should": [{"match": {"assignee_names": name}} for name in query.assignees],
                    "minimum_should_match": 1,
                }
            }
        )
    if query.before_date:
        cutoff = query.before_date.isoformat()
        filters.append(
            {
                "bool": {
                    "should": [
                        {"range": {"priority_date": {"lt": cutoff}}},
                        {"range": {"filing_date": {"lt": cutoff}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    if query.after_date:
        cutoff = query.after_date.isoformat()
        filters.append(
            {
                "bool": {
                    "should": [
                        {"range": {"priority_date": {"gt": cutoff}}},
                        {"range": {"filing_date": {"gt": cutoff}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return filters


def build_es_must_not(query: SearchQuery) -> list[dict]:
    """Translate excluded keywords and denied numbers into must_not clauses."""
    search_fields = es_search_fields(query.fields)
    must_not: list[dict] = []
    for keyword in query.excluded_keywords:
        must_not.append(
            {
                "multi_match": {
                    "query": keyword,
                    "fields": search_fields,
                    "type": "best_fields",
                    "operator": "and",
                }
            }
        )
    for number in query.exclude_numbers:
        must_not.append({"term": {"patent_number_norm": normalized_number(number)}})
    return must_not


def build_es_query(query: SearchQuery) -> dict:
    """Translate the full :class:`SearchQuery` parameter set into an ES bool query.

    - ``keywords`` -> a nested bool of per-keyword (best_fields + boosted
      phrase) ``should`` clauses gated by ``minimum_should_match`` from
      :meth:`SearchQuery.effective_minimum_match`.
    - ``text`` -> a fuzzy best_fields + phrase clause pair (at least one must hit).
    - ``required_keywords`` -> ``must`` clauses (AND semantics).
    - ``excluded_keywords`` / ``exclude_numbers`` -> ``must_not``.
    - everything else -> ``filter`` via :func:`build_es_filters`.
    """
    search_fields = es_search_fields(query.fields)
    must: list[dict] = []

    if query.keywords:
        should: list[dict] = []
        for keyword in query.keywords:
            should.extend(_keyword_clauses(keyword, search_fields))
        must.append(
            {
                "bool": {
                    "should": should,
                    "minimum_should_match": query.effective_minimum_match(),
                }
            }
        )

    if query.text:
        must.append(
            {
                "bool": {
                    "should": _keyword_clauses(query.text, search_fields),
                    "minimum_should_match": 1,
                }
            }
        )

    for keyword in query.required_keywords:
        must.append(
            {
                "bool": {
                    "should": _keyword_clauses(keyword, search_fields),
                    "minimum_should_match": 1,
                }
            }
        )

    if not must:
        must.append({"match_all": {}})

    bool_query: dict[str, Any] = {"must": must}
    must_not = build_es_must_not(query)
    if must_not:
        bool_query["must_not"] = must_not
    filters = build_es_filters(query)
    if filters:
        bool_query["filter"] = filters
    return {"bool": bool_query}


def build_knn_query(vector: list[float], limit: int, query: SearchQuery | None = None,
                    *, num_candidates: int | None = None) -> dict:
    """Build a nested-chunk kNN clause with SearchQuery metadata filters."""
    knn: dict[str, Any] = {
        "field": "chunks.embedding",
        "query_vector": vector,
        "k": limit,
        "num_candidates": num_candidates or max(100, limit * 10),
        "inner_hits": {"size": 1, "_source": ["chunks.text"]},
    }
    if query is not None:
        filters = build_es_filters(query)
        must_not = build_es_must_not(query)
        if filters or must_not:
            filter_bool: dict[str, Any] = {}
            if filters:
                filter_bool["filter"] = filters
            if must_not:
                filter_bool["must_not"] = must_not
            knn["filter"] = {"bool": filter_bool}
    return knn


def build_es_mapping(embedding_dims: int | None = None) -> dict:
    """Index settings + mappings for the patentkit patents index.

    Text fields for title/abstract/specification/claims, keyword fields for
    numbers/CPC/inventors/assignees/country, date fields, and (when
    ``embedding_dims`` is given) nested chunks with a cosine dense_vector.
    """
    properties: dict[str, Any] = {
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
        "abstract": {"type": "text"},
        "specification": {"type": "text"},
        "claims_text": {"type": "text"},
        "patent_number_norm": {"type": "keyword"},
        "number": {"type": "keyword"},
        "country": {"type": "keyword"},
        "kind": {"type": "keyword"},
        "cpc": {"type": "keyword"},
        "inventor_names": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "assignee_names": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "priority_date": {"type": "date"},
        "filing_date": {"type": "date"},
        "publication_date": {"type": "date"},
        "grant_date": {"type": "date"},
        "expiration_date": {"type": "date"},
        "status": {"type": "keyword"},
    }
    if embedding_dims:
        properties["chunks"] = {
            "type": "nested",
            "properties": {
                "chunk_number": {"type": "integer"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        }
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"dynamic": False, "properties": properties},
    }


def patent_to_doc(patent: Patent, embeddings: EmbeddingProvider | None = None) -> dict:
    """Serialize a canonical Patent into an ES document.

    The full ``model_dump(mode="json")`` payload is kept in ``_source`` (so
    documents round-trip back into :class:`Patent`); flattened search fields
    are added on top. When ``embeddings`` is given the specification is
    chunked and embedded into the nested ``chunks`` field.
    """
    doc = patent.model_dump(mode="json")
    doc["patent_number_norm"] = normalized_number(patent.patent_number)
    doc["number"] = patent.patent_number.number
    doc["country"] = patent.patent_number.country_code
    doc["kind"] = patent.patent_number.kind_code
    doc["claims_text"] = "\n".join(c.text for c in patent.claims)
    doc["cpc"] = [c.code for c in patent.classifications]
    doc["inventor_names"] = [i.name for i in patent.inventors]
    doc["assignee_names"] = [a.name for a in patent.assignees]
    if embeddings is not None:
        text = patent.specification or patent.text_for_search()
        texts = chunk_text(text)
        if texts:
            vectors = embeddings.embed(texts)
            doc["chunks"] = [
                {"chunk_number": i, "text": chunk, "embedding": vector}
                for i, (chunk, vector) in enumerate(zip(texts, vectors))
            ]
    return doc


class ElasticsearchStore:
    """Elasticsearch-backed store implementing KeywordStore and VectorStore.

    Args:
        index: index name (default ``"patents"``).
        host: cluster URL; falls back to the ``ELASTICSEARCH_HOST`` env var.
        api_key: cluster API key; falls back to ``ELASTICSEARCH_API_KEY``.
        client: a pre-built ``elasticsearch.Elasticsearch`` (or compatible)
            client; when given, host/api_key resolution is skipped entirely.
        embeddings: optional :class:`EmbeddingProvider`; enables the
            VectorStore side (chunk embedding at index time, kNN at query time).
    """

    def __init__(
        self,
        index: str = "patents",
        host: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        embeddings: EmbeddingProvider | None = None,
    ):
        self.index_name = index
        self.embeddings = embeddings
        self._client = client
        if client is None:
            self._host = resolve_key("ELASTICSEARCH_HOST", host)
            self._api_key = resolve_key("ELASTICSEARCH_API_KEY", api_key, required=False)
        else:
            self._host = host
            self._api_key = api_key

    @property
    def client(self) -> Any:
        """The Elasticsearch client, created lazily on first use."""
        if self._client is None:
            elasticsearch = _import_elasticsearch()
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = elasticsearch.Elasticsearch(self._host, **kwargs)
        return self._client

    def _embedding_dims(self) -> int | None:
        if self.embeddings is None:
            return None
        dims = getattr(self.embeddings, "dimensions", None)
        if dims:
            return int(dims)
        return len(self.embeddings.embed(["dimension probe"])[0])

    def ensure_index(self) -> bool:
        """Create the index with the patentkit mapping if it does not exist.

        Returns True if the index was created, False if it already existed.
        """
        if self.client.indices.exists(index=self.index_name):
            return False
        body = build_es_mapping(self._embedding_dims())
        self.client.indices.create(
            index=self.index_name, settings=body["settings"], mappings=body["mappings"]
        )
        logger.info("Created Elasticsearch index %r", self.index_name)
        return True

    # -- KeywordStore / VectorStore: indexing --------------------------------

    def index_patents(self, patents: Iterable[Patent]) -> int:
        """Bulk-index canonical patents; returns the count indexed."""
        elasticsearch = _import_elasticsearch()
        from elasticsearch import helpers  # noqa: F401  (lazy, optional extra)

        actions = []
        for patent in patents:
            doc = patent_to_doc(patent, self.embeddings)
            actions.append(
                {"_index": self.index_name, "_id": doc["patent_number_norm"], "_source": doc}
            )
        if not actions:
            return 0
        success, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        if errors:
            logger.error("Bulk indexing reported %d errors: %s", len(errors), errors[:3])
        logger.info("Indexed %d patents into %r", success, self.index_name)
        return success

    def index(self, patents: Iterable[Patent]) -> int:  # type: ignore[override]
        """KeywordStore/VectorStore protocol method — alias of :meth:`index_patents`."""
        return self.index_patents(patents)

    def __len__(self) -> int:
        return int(self.client.count(index=self.index_name)["count"])

    # -- KeywordStore: search -------------------------------------------------

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Full-parameter keyword search; see :func:`build_es_query`."""
        highlight_fields = {_ES_FIELD_NAMES.get(f, f): {} for f in query.fields}
        body = {
            "query": build_es_query(query),
            "size": query.limit,
            # max_analyzed_offset: real specifications can exceed the index's
            # 1M-char highlight analysis limit; truncate instead of erroring.
            "highlight": {"fields": highlight_fields, "max_analyzed_offset": 999_999},
        }
        response = self.client.search(index=self.index_name, body=body)
        return [self._hit_to_result(hit, "elasticsearch bm25") for hit in response["hits"]["hits"]]

    # -- VectorStore: search ---------------------------------------------------

    def search_text(self, text: str, *, limit: int = 50,
                    query: SearchQuery | None = None) -> list[SearchResult]:
        """Embed ``text`` and run kNN over nested chunk vectors with filters."""
        if self.embeddings is None:
            raise RuntimeError(
                "ElasticsearchStore was built without an embedding provider; "
                "pass embeddings=... to enable vector search."
            )
        [vector] = self.embeddings.embed([text])
        body = {"knn": build_knn_query(vector, limit, query), "size": limit}
        response = self.client.search(index=self.index_name, body=body)
        return [
            self._hit_to_result(hit, f"elasticsearch knn ({self.embeddings.model_name})")
            for hit in response["hits"]["hits"]
        ]

    def get(self, number: PatentNumber) -> Optional[Patent]:
        """Fetch one patent by (kind-insensitive) number."""
        body = {
            "query": {"term": {"patent_number_norm": normalized_number(number)}},
            "size": 1,
        }
        response = self.client.search(index=self.index_name, body=body)
        hits = response["hits"]["hits"]
        if not hits:
            return None
        return Patent.model_validate(self._strip_doc(hits[0]["_source"]))

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _strip_doc(source: dict) -> dict:
        """Drop the flattened search-only fields before Patent validation."""
        return {
            k: v
            for k, v in source.items()
            if k not in (
                "patent_number_norm", "number", "country", "kind", "claims_text",
                "cpc", "inventor_names", "assignee_names", "chunks",
            )
        }

    def _hit_to_result(self, hit: dict, explanation: str) -> SearchResult:
        patent = Patent.model_validate(self._strip_doc(hit["_source"]))
        score = float(hit.get("_score") or 0.0)
        passages: list[Passage] = []
        for es_field, fragments in (hit.get("highlight") or {}).items():
            field_name = _ES_FIELD_NAMES_REVERSE.get(es_field, es_field)
            for fragment in fragments:
                passages.append(Passage(text=fragment, field=field_name, score=score))
        inner = hit.get("inner_hits") or {}
        for inner_hits in inner.values():
            for inner_hit in inner_hits.get("hits", {}).get("hits", []):
                chunk_source = inner_hit.get("_source") or {}
                text = chunk_source.get("text")
                if text:
                    passages.append(Passage(
                        text=text[:400], field="specification",
                        score=float(inner_hit.get("_score") or 0.0),
                    ))
        return SearchResult(
            patent_number=patent.patent_number,
            score=score,
            patent=patent,
            passages=passages,
            explanation=explanation,
        )
