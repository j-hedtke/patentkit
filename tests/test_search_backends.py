"""Tests for the Elasticsearch query builders and hybrid fusion.

No network and no elasticsearch package needed: query construction is pure,
and the hybrid searcher runs over the in-memory BM25 + vector stores.
"""

from __future__ import annotations

from datetime import date

import pytest

from patentkit.models import Classification, Patent, PatentNumber
from patentkit.search.base import Passage, SearchQuery, SearchResult
from patentkit.search.bm25 import BM25Store
from patentkit.search.elasticsearch_store import (
    build_es_query,
    build_knn_query,
    normalized_number,
    patent_to_doc,
)
from patentkit.search.hybrid import HybridSearcher, rrf_fuse, zscore_combine
from patentkit.search.vector import HashingEmbeddings, InMemoryVectorStore


def make_patent(number: str, title: str, abstract: str = "", spec: str = "",
                cpc: list[str] | None = None) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(number),
        title=title,
        abstract=abstract,
        specification=spec,
        classifications=[Classification(code=c) for c in (cpc or [])],
    )


# ---------------------------------------------------------------------------
# build_es_query translation
# ---------------------------------------------------------------------------


class TestBuildEsQuery:
    def test_keywords_minimum_match_default(self):
        keywords = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        query = SearchQuery(keywords=keywords)
        body = build_es_query(query)
        keyword_bool = body["bool"]["must"][0]["bool"]
        # default = len(keywords) // 3
        assert keyword_bool["minimum_should_match"] == 2
        # best_fields + phrase clause per keyword
        assert len(keyword_bool["should"]) == 12
        phrase_clauses = [
            c for c in keyword_bool["should"] if c["multi_match"].get("type") == "phrase"
        ]
        assert len(phrase_clauses) == 6
        assert all(c["multi_match"]["boost"] == 2.0 for c in phrase_clauses)
        best_fields = [
            c for c in keyword_bool["should"] if c["multi_match"]["type"] == "best_fields"
        ]
        assert all(c["multi_match"]["fuzziness"] == "AUTO" for c in best_fields)

    def test_explicit_minimum_match(self):
        query = SearchQuery(keywords=["a", "b", "c"], minimum_match=3)
        body = build_es_query(query)
        assert body["bool"]["must"][0]["bool"]["minimum_should_match"] == 3

    def test_excluded_keywords_become_must_not(self):
        query = SearchQuery(keywords=["widget"], excluded_keywords=["asbestos"])
        body = build_es_query(query)
        must_not = body["bool"]["must_not"]
        assert any(
            clause.get("multi_match", {}).get("query") == "asbestos" for clause in must_not
        )

    def test_excluded_numbers_become_must_not_terms(self):
        query = SearchQuery(exclude_numbers=[PatentNumber.parse("US7,654,321 B2")])
        body = build_es_query(query)
        assert {"term": {"patent_number_norm": "US7654321"}} in body["bool"]["must_not"]

    def test_art_class_prefix_filter(self):
        query = SearchQuery(art_classes=["G06F16", "H04L"])
        body = build_es_query(query)
        prefix_filter = next(
            f for f in body["bool"]["filter"] if "bool" in f and
            any("prefix" in clause for clause in f["bool"]["should"])
        )
        prefixes = [clause["prefix"]["cpc"] for clause in prefix_filter["bool"]["should"]]
        assert prefixes == ["G06F16", "H04L"]
        assert prefix_filter["bool"]["minimum_should_match"] == 1

    def test_before_date_filter_on_priority_or_filing(self):
        query = SearchQuery(before_date=date(2005, 1, 1))
        body = build_es_query(query)
        date_filter = body["bool"]["filter"][0]["bool"]
        ranges = {list(c["range"])[0]: c["range"] for c in date_filter["should"]}
        assert ranges["priority_date"]["priority_date"]["lt"] == "2005-01-01"
        assert ranges["filing_date"]["filing_date"]["lt"] == "2005-01-01"

    def test_after_date_filter(self):
        query = SearchQuery(after_date=date(1999, 12, 31))
        body = build_es_query(query)
        date_filter = body["bool"]["filter"][0]["bool"]
        assert any(
            c["range"].get("priority_date", {}).get("gt") == "1999-12-31"
            for c in date_filter["should"]
        )

    def test_required_keywords_become_must(self):
        query = SearchQuery(keywords=["foo"], required_keywords=["bar baz"])
        body = build_es_query(query)
        # must[0] is the keywords bool, must[1] the required keyword
        required = body["bool"]["must"][1]["bool"]
        assert required["minimum_should_match"] == 1
        assert required["should"][0]["multi_match"]["query"] == "bar baz"

    def test_countries_and_include_numbers(self):
        query = SearchQuery(
            countries=["US", "EP"],
            include_numbers=[PatentNumber.parse("US0123456")],
        )
        body = build_es_query(query)
        assert {"terms": {"country": ["US", "EP"]}} in body["bool"]["filter"]
        assert {"terms": {"patent_number_norm": ["US123456"]}} in body["bool"]["filter"]

    def test_empty_query_is_match_all(self):
        body = build_es_query(SearchQuery())
        assert body["bool"]["must"] == [{"match_all": {}}]
        assert "must_not" not in body["bool"]
        assert "filter" not in body["bool"]

    def test_fields_subset_maps_claims_to_claims_text(self):
        query = SearchQuery(keywords=["foo"], fields=["title", "claims"])
        body = build_es_query(query)
        fields = body["bool"]["must"][0]["bool"]["should"][0]["multi_match"]["fields"]
        assert fields == ["title^2.0", "claims_text^1.5"]


class TestKnnAndDoc:
    def test_knn_query_carries_filters(self):
        query = SearchQuery(countries=["US"], excluded_keywords=["asbestos"])
        knn = build_knn_query([0.1, 0.2], limit=7, query=query)
        assert knn["field"] == "chunks.embedding"
        assert knn["k"] == 7
        assert knn["num_candidates"] >= 70
        assert {"terms": {"country": ["US"]}} in knn["filter"]["bool"]["filter"]
        assert any(
            c.get("multi_match", {}).get("query") == "asbestos"
            for c in knn["filter"]["bool"]["must_not"]
        )

    def test_knn_query_without_filters(self):
        knn = build_knn_query([0.5], limit=3)
        assert "filter" not in knn

    def test_patent_to_doc_flattens_and_chunks(self):
        patent = make_patent("US1234567B1", "Widget", abstract="An abstract.",
                             spec="word " * 50, cpc=["G06F16/00"])
        doc = patent_to_doc(patent, embeddings=HashingEmbeddings(dimensions=16))
        assert doc["patent_number_norm"] == "US1234567"
        assert doc["country"] == "US"
        assert doc["cpc"] == ["G06F16/00"]
        assert doc["chunks"] and len(doc["chunks"][0]["embedding"]) == 16
        assert doc["title"] == "Widget"

    def test_normalized_number_kind_insensitive(self):
        assert normalized_number(PatentNumber.parse("US07654321B2")) == "US7654321"
        assert normalized_number(PatentNumber.parse("US7654321")) == "US7654321"


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def result(number: str, score: float, patent: Patent | None = None,
           passages: list[Passage] | None = None) -> SearchResult:
    return SearchResult(
        patent_number=PatentNumber.parse(number), score=score,
        patent=patent, passages=passages or [],
    )


class TestRrfFuse:
    def test_ordering_matches_hand_computation(self):
        list_a = [result("US1", 9.0), result("US2", 5.0)]
        list_b = [result("US2", 0.9), result("US3", 0.5)]
        fused = rrf_fuse([list_a, list_b], k=60)
        numbers = [str(r.patent_number) for r in fused]
        assert numbers == ["US2", "US1", "US3"]
        # US2: rank 1 in A (1/62) + rank 0 in B (1/61)
        assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
        assert fused[1].score == pytest.approx(1 / 61)
        assert fused[2].score == pytest.approx(1 / 62)

    def test_kind_codes_fuse_to_same_document(self):
        fused = rrf_fuse([[result("US7654321B2", 1.0)], [result("US7654321", 1.0)]])
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2 / 61)

    def test_passages_merged_and_patent_kept(self):
        patent = make_patent("US2", "Kept")
        a = result("US2", 5.0, patent=None,
                   passages=[Passage(text="from keyword", field="title")])
        b = result("US2", 0.9, patent=patent,
                   passages=[Passage(text="from vector", field="specification"),
                             Passage(text="from keyword", field="title")])
        [fused] = rrf_fuse([[a], [b]])
        assert fused.patent is patent
        texts = sorted(p.text for p in fused.passages)
        assert texts == ["from keyword", "from vector"]  # deduped

    def test_empty_input(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []


class TestZscoreCombine:
    def test_opposite_rankings_cancel(self):
        combined = zscore_combine({"a": [1.0, 2.0, 3.0], "b": [30.0, 20.0, 10.0]})
        assert combined == pytest.approx([0.0, 0.0, 0.0])

    def test_weighted(self):
        combined = zscore_combine(
            {"a": [0.0, 1.0], "b": [0.0, 1.0]}, weights={"a": 1.0, "b": 3.0}
        )
        assert combined[1] > combined[0]
        assert combined[1] == pytest.approx(4.0)  # z=+1 each, weights 1+3

    def test_misaligned_lengths_raise(self):
        with pytest.raises(ValueError):
            zscore_combine({"a": [1.0], "b": [1.0, 2.0]})


# ---------------------------------------------------------------------------
# HybridSearcher over in-memory stores
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus() -> list[Patent]:
    return [
        make_patent(
            "US1000001", "Neural network training system",
            abstract="Training deep neural networks with gradient descent.",
            spec="A neural network is trained using backpropagation and gradient descent. "
                 "The neural network layers learn weights from labeled examples. " * 4,
        ),
        make_patent(
            "US1000002", "Gear shaft assembly",
            abstract="A gear shaft with bearings for torque transmission.",
            spec="The gear shaft transmits torque through helical gears and bearings. " * 4,
        ),
        make_patent(
            "US1000003", "Neural network inference accelerator",
            abstract="Hardware accelerator for neural network inference.",
            spec="The accelerator executes neural network inference with matrix units. " * 4,
        ),
        make_patent(
            "US1000004", "Polymer coating composition",
            abstract="A chemical polymer coating for corrosion resistance.",
            spec="The polymer composition includes epoxy resins and curing agents. " * 4,
        ),
    ]


class TestHybridSearcher:
    def test_end_to_end_fusion(self, corpus):
        keyword_store = BM25Store()
        keyword_store.index(corpus)
        vector_store = InMemoryVectorStore(HashingEmbeddings(dimensions=128))
        vector_store.index(corpus)

        searcher = HybridSearcher(keyword_store, vector_store, k=60)
        query = SearchQuery(
            keywords=["neural network"], text="training neural networks", limit=3
        )
        results = searcher.search(query)

        assert results, "hybrid search returned nothing"
        assert len(results) <= 3
        top = str(results[0].patent_number)
        assert top in ("US1000001", "US1000003")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # a document found by both legs carries a fused rrf explanation
        assert results[0].explanation and results[0].explanation.startswith("rrf(k=60)")
        assert results[0].patent is not None

    def test_falls_back_to_keywords_for_vector_text(self, corpus):
        keyword_store = BM25Store()
        keyword_store.index(corpus)
        vector_store = InMemoryVectorStore(HashingEmbeddings(dimensions=128))
        vector_store.index(corpus)
        searcher = HybridSearcher(keyword_store, vector_store)
        # no query.text: vector leg uses " ".join(keywords)
        results = searcher.search(SearchQuery(keywords=["gear shaft torque"], limit=2))
        assert str(results[0].patent_number) == "US1000002"

    def test_exclusions_apply_to_both_legs(self, corpus):
        keyword_store = BM25Store()
        keyword_store.index(corpus)
        vector_store = InMemoryVectorStore(HashingEmbeddings(dimensions=128))
        vector_store.index(corpus)
        searcher = HybridSearcher(keyword_store, vector_store)
        query = SearchQuery(
            keywords=["neural network"],
            exclude_numbers=[PatentNumber.parse("US1000001")],
            limit=4,
        )
        numbers = {str(r.patent_number) for r in searcher.search(query)}
        assert "US1000001" not in numbers
