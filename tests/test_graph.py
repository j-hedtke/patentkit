"""Tests for the gradual concept graph: stores, harvesting, staged
promotion, expansion, and the guided-search integration."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from patentkit.agents import GuidedSearch, ResultFeedback, SearchFeedback
from patentkit.graph import (
    STAGE_CANDIDATE_CLUSTER,
    STAGE_PRODUCTION,
    STAGE_REVIEWED_CANONICAL,
    ConceptGraph,
    ConceptNode,
    MatchPair,
    MatchPairStore,
    expand_limitation,
    harvest_match_pairs,
    promote,
    reject_alias,
    review,
)
from patentkit.models import Claim, Patent, PatentNumber
from patentkit.search.bm25 import BM25Store
from tests.fakes import FakeLLM


# ----------------------------------------------------------------- helpers

def make_pair(phrase: str, patent: str, search: str, outcome: str = "accepted",
              matched: str = "the matched passage") -> MatchPair:
    return MatchPair(query_limitation=phrase, matched_text=matched,
                     patent_id=patent, search_id=search, outcome=outcome,
                     created_at="2026-06-10")


def seeded_store(tmp_path, pairs) -> MatchPairStore:
    store = MatchPairStore(tmp_path / "graph")
    for pair in pairs:
        store.add(pair)
    return store


PHRASE = "rank documents by embedding similarity"
VARIANT_A = "ranking documents by embedding similarity scores"
VARIANT_B = "rank documents using embedding similarity"


def happy_pairs(patents=("US1A", "US2A", "US3A")) -> list[MatchPair]:
    """5 distinct searches, 6 accepted pairs, 3 phrase variants."""
    p = list(patents)
    return [
        make_pair(PHRASE, p[0], "s1"),
        make_pair(PHRASE, p[1], "s2"),
        make_pair(VARIANT_A, p[2 % len(p)], "s3"),
        make_pair(VARIANT_B, p[0], "s4"),
        make_pair(PHRASE, p[1], "s5"),
        make_pair(VARIANT_A, p[2 % len(p)], "s5"),
    ]


# ------------------------------------------------------------------ stores

def test_match_pair_store_round_trip(tmp_path) -> None:
    store = seeded_store(tmp_path, [
        make_pair(PHRASE, "US1A", "s1"),
        make_pair(VARIANT_A, "US2A", "s2", outcome="unreviewed"),
    ])
    # a fresh instance over the same root sees the same JSONL rows
    fresh = MatchPairStore(tmp_path / "graph")
    pairs = list(fresh.iter())
    assert len(fresh) == 2
    assert pairs[0].query_limitation == PHRASE
    assert pairs[0].created_at == "2026-06-10"
    assert [p.patent_id for p in fresh.filter(outcome="accepted")] == ["US1A"]
    assert fresh.filter(patent_id="US2A", search_id="s2")[0].outcome == "unreviewed"
    assert fresh.filter(patent_id="nope") == []

    # mark_outcome rewrites matching rows (scoped by search_id)
    updated = fresh.mark_outcome("US2A", "accepted", search_id="s2",
                                 feedback_type="teaches_limitation")
    assert updated == 1
    again = MatchPairStore(tmp_path / "graph").filter(patent_id="US2A")[0]
    assert again.outcome == "accepted"
    assert again.feedback_type == "teaches_limitation"
    assert fresh.mark_outcome("US2A", "rejected", search_id="other") == 0


def test_concept_graph_round_trip(tmp_path) -> None:
    graph = ConceptGraph(tmp_path / "graph")
    node = graph.add(ConceptNode(
        canonical_name="RANK_DOCUMENTS_BY_EMBEDDING_SIMILARITY",
        aliases=[PHRASE, VARIANT_B], stage=STAGE_REVIEWED_CANONICAL,
        evidence={"searches": 5, "accepted_charts": 6, "rejected": 0,
                  "patents": ["US1A"]}))
    graph.save()

    fresh = ConceptGraph(tmp_path / "graph").load()
    assert len(fresh) == 1
    loaded = fresh.get("RANK_DOCUMENTS_BY_EMBEDDING_SIMILARITY")
    assert loaded == node
    # alias lookup is case/punctuation-insensitive
    assert fresh.find_by_alias("Rank Documents, using Embedding Similarity!") is loaded
    assert fresh.find_by_alias("rank documents by embedding similarity") is loaded
    assert fresh.find_by_alias("totally unrelated phrase") is None
    # loading a missing file is a no-op
    assert len(ConceptGraph(tmp_path / "empty").load()) == 0


# -------------------------------------------------------------- harvesting

def test_harvest_from_result_rows() -> None:
    rows = [
        {"patent_number": "US1A", "title": "t", "score": 0.9,
         "passages": [{"text": "a capacitive probe drives the controller",
                       "field": "claims", "score": 0.9}],
         "why": "probe art"},
        {"name": "SmartSprinkler",  # infringement rows use product names
         "passages": [], "why": "practices every limitation"},
        {"passages": [{"text": "ignored — no identifier"}]},
    ]

    class DuckLimitation:  # the concurrency contract: only .text is assumed
        text = "scheduling watering from moisture readings"

    pairs = harvest_match_pairs(rows, limitations=["soil moisture sensing",
                                                   DuckLimitation()],
                                search_id="sess1", created_at="2026-06-10")
    # 2 usable rows x 1 text each x 2 limitations
    assert len(pairs) == 4
    assert all(p.outcome == "unreviewed" for p in pairs)
    assert all(p.search_id == "sess1" and p.created_at == "2026-06-10" for p in pairs)
    by_patent = {p.patent_id for p in pairs}
    assert by_patent == {"US1A", "SmartSprinkler"}
    first = [p for p in pairs if p.patent_id == "US1A"][0]
    assert first.matched_text == "a capacitive probe drives the controller"
    assert first.section == "claims"
    assert {p.query_limitation for p in pairs} == {
        "soil moisture sensing", "scheduling watering from moisture readings"}
    # rows without passages fall back to the agent's "why"
    sprinkler = [p for p in pairs if p.patent_id == "SmartSprinkler"][0]
    assert sprinkler.matched_text == "practices every limitation"
    assert sprinkler.section == "agent_rationale"


def test_harvest_from_claim_chart_shaped_object() -> None:
    """ClaimChart stand-in built from SimpleNamespace — no analysis imports."""
    lim_a = SimpleNamespace(text="wireless soil moisture sensor nodes")
    lim_b = SimpleNamespace(text="controller scheduling watering of zones")
    chart = SimpleNamespace(
        query_patent="US8123456B2", claim_number=1, limitations=[lim_a, lim_b],
        references=[
            SimpleNamespace(
                reference_number="US7000001B1",
                findings=[
                    SimpleNamespace(limitation=lim_a, status="disclosed",
                                    quotes=["sensor nodes transmit readings",
                                            "a wireless mesh network"],
                                    citation="col. 3, ll. 45-52"),
                    SimpleNamespace(limitation=lim_b, status="not_disclosed",
                                    quotes=[], citation=None),
                ]),
            SimpleNamespace(
                reference_number="US7000002B1",
                findings=[
                    SimpleNamespace(limitation=lim_b, status="partial",
                                    quotes=["watering is adjusted"], citation=None),
                ]),
        ])

    pairs = harvest_match_pairs(chart, search_id="chart1", created_at="2026-06-10")
    assert len(pairs) == 4  # 2 quotes + 1 empty rejected + 1 partial

    accepted = [p for p in pairs if p.outcome == "accepted"]
    assert len(accepted) == 2
    assert all(p.patent_id == "US7000001B1" for p in accepted)
    assert all(p.query_limitation == lim_a.text for p in accepted)
    assert all(p.feedback_type == "teaches_limitation" for p in accepted)
    assert accepted[0].matched_text == "sensor nodes transmit readings"
    assert accepted[0].section == "col. 3, ll. 45-52"

    rejected = [p for p in pairs if p.outcome == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].query_limitation == lim_b.text
    assert rejected[0].matched_text == ""  # the rejection itself is the signal

    partial = [p for p in pairs if p.outcome == "unreviewed"]
    assert len(partial) == 1 and partial[0].patent_id == "US7000002B1"


def test_harvest_dispatch_trace_and_result_model() -> None:
    trace = SimpleNamespace(shortlist_history=[
        [{"number": "US1A", "why": "early angle"}],
        [{"number": "US1A", "why": "mesh network nodes",
          "key_passage": "nodes feed a controller"}],
    ])
    pairs = harvest_match_pairs(trace, limitations=["sensor mesh"], search_id="s9")
    assert len(pairs) == 1  # only the FINAL shortlist snapshot
    assert pairs[0].patent_id == "US1A"
    assert pairs[0].matched_text == "nodes feed a controller"
    assert pairs[0].section == "shortlist"

    result_model = SimpleNamespace(results=[
        {"patent_number": "US2A", "passages": ["verbatim passage"], "why": "w"}])
    pairs = harvest_match_pairs(result_model, limitations=["sensor mesh"])
    assert len(pairs) == 1 and pairs[0].matched_text == "verbatim passage"

    with pytest.raises(TypeError):
        harvest_match_pairs(object())


# --------------------------------------------------------------- promotion

def test_promotion_happy_path_caps_at_stage_3(tmp_path) -> None:
    store = seeded_store(tmp_path, happy_pairs())
    graph = ConceptGraph(tmp_path / "graph")
    nodes = promote(graph, store, min_searches=5, min_accepted=3,
                    max_reject_rate=0.2, min_patents=3)

    assert len(nodes) == 1, "phrase variants must cluster into ONE concept"
    node = nodes[0]
    assert node.stage == STAGE_REVIEWED_CANONICAL  # never auto-promoted past 3
    assert node.canonical_name == "RANK_DOCUMENTS_BY_EMBEDDING_SIMILARITY"
    assert set(node.aliases) == {PHRASE, VARIANT_A, VARIANT_B}
    assert node.evidence["searches"] == 5
    assert node.evidence["accepted_charts"] == 6
    assert node.evidence["rejected"] == 0
    assert sorted(node.evidence["patents"]) == ["US1A", "US2A", "US3A"]
    # promote saved the graph (the residue is durable)
    assert ConceptGraph(tmp_path / "graph").load().find_by_alias(PHRASE) is not None
    # idempotent: re-running changes nothing
    again = promote(graph, store, min_searches=5, min_accepted=3,
                    max_reject_rate=0.2, min_patents=3)
    assert len(again) == 1 and again[0].stage == STAGE_REVIEWED_CANONICAL
    assert len(graph) == 1


def test_promotion_min_patents_gate(tmp_path) -> None:
    # 5 searches / 6 accepted but only 2 distinct patents -> stays stage 2
    store = seeded_store(tmp_path, happy_pairs(patents=("US1A", "US2A")))
    graph = ConceptGraph(tmp_path / "graph")
    (node,) = promote(graph, store, min_searches=5, min_accepted=3,
                      max_reject_rate=0.2, min_patents=3)
    assert node.stage == STAGE_CANDIDATE_CLUSTER
    assert sorted(node.evidence["patents"]) == ["US1A", "US2A"]


def test_promotion_reject_rate_gate(tmp_path) -> None:
    pairs = happy_pairs() + [
        make_pair(PHRASE, "US4A", "s6", outcome="rejected"),
        make_pair(VARIANT_B, "US5A", "s7", outcome="rejected"),
    ]
    store = seeded_store(tmp_path, pairs)
    graph = ConceptGraph(tmp_path / "graph")
    (node,) = promote(graph, store, min_searches=5, min_accepted=3,
                      max_reject_rate=0.2, min_patents=3)
    assert node.evidence["rejected"] == 2  # 2/(6+2) = 0.25 > 0.2
    assert node.stage == STAGE_CANDIDATE_CLUSTER


def test_promotion_ignores_unreviewed_and_blank_limitations(tmp_path) -> None:
    store = seeded_store(tmp_path, [
        make_pair(PHRASE, "US1A", "s1", outcome="unreviewed"),
        make_pair("", "US2A", "s2", outcome="accepted"),
    ])
    graph = ConceptGraph(tmp_path / "graph")
    assert promote(graph, store) == []
    assert len(graph) == 0


def test_stage_4_requires_human_review(tmp_path) -> None:
    store = seeded_store(tmp_path, happy_pairs())
    graph = ConceptGraph(tmp_path / "graph")
    (node,) = promote(graph, store)
    assert node.stage == STAGE_REVIEWED_CANONICAL

    # only review(approved=True) reaches production
    review(node, approved=True)
    assert node.stage == STAGE_PRODUCTION
    # promote never lowers (or re-raises questions about) a reviewed node
    (node,) = promote(graph, store)
    assert node.stage == STAGE_PRODUCTION

    # an unready node cannot be approved
    young = ConceptNode(canonical_name="YOUNG", aliases=["young phrase"], stage=1)
    with pytest.raises(ValueError, match="stage"):
        review(young, approved=True)

    # a rejecting review demotes back to candidate cluster
    review(node, approved=False)
    assert node.stage == STAGE_CANDIDATE_CLUSTER


def test_reject_alias_removes_bad_synonym_for_good(tmp_path) -> None:
    store = seeded_store(tmp_path, happy_pairs())
    graph = ConceptGraph(tmp_path / "graph")
    (node,) = promote(graph, store)
    assert VARIANT_A in node.aliases

    reject_alias(node, VARIANT_A)
    assert VARIANT_A not in node.aliases
    assert VARIANT_A in node.evidence["rejected_aliases"]
    # promotion never re-adds a rejected alias
    (node,) = promote(graph, store)
    assert VARIANT_A not in node.aliases
    # and a node reduced below an alias pair drops to stage 1
    reject_alias(node, VARIANT_B)
    reject_alias(node, PHRASE)  # no-op guard at 1 alias? remove down to 0
    assert node.stage <= STAGE_CANDIDATE_CLUSTER


# --------------------------------------------------------------- expansion

def test_expand_limitation_respects_min_stage() -> None:
    graph = ConceptGraph("unused-root")
    graph.add(ConceptNode(
        canonical_name="WIRELESS_SOIL_MOISTURE_SENSING",
        aliases=["wireless soil moisture sensor nodes",
                 "soil humidity radio sensors",
                 "wireless moisture probe network"],
        stage=STAGE_REVIEWED_CANONICAL))
    graph.add(ConceptNode(
        canonical_name="ZONE_WATERING_SCHEDULER",
        aliases=["controller scheduling watering of zones",
                 "zone irrigation timer"],
        stage=STAGE_CANDIDATE_CLUSTER))  # below the default min_stage

    # exact alias hit -> the OTHER aliases, input phrasing excluded
    out = expand_limitation(graph, "wireless soil moisture sensor nodes")
    assert out == ["soil humidity radio sensors", "wireless moisture probe network"]
    # duck-typed limitation objects work too
    duck = SimpleNamespace(text="Wireless soil-moisture SENSOR nodes")
    assert expand_limitation(graph, duck) == out
    # alias contained in a longer limitation matches as well
    longer = ("a plurality of wireless soil moisture sensor nodes transmitting "
              "readings to a controller")
    assert "soil humidity radio sensors" in expand_limitation(graph, longer)

    # stage-2 concepts stay silent at the default min_stage=3 ...
    assert expand_limitation(graph, "controller scheduling watering of zones") == []
    # ... but surface when the caller lowers the bar
    assert expand_limitation(graph, "controller scheduling watering of zones",
                             min_stage=2) == ["zone irrigation timer"]
    # no match -> no expansion; blank input -> no expansion
    assert expand_limitation(graph, "quantum entanglement distillation") == []
    assert expand_limitation(graph, "   ") == []


# ------------------------------------------------------- guided integration

def make_patent(number: str, title: str, claim: str, spec: str,
                priority: date) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(number), title=title,
        abstract=spec[:150], claims=[Claim(number=1, text=claim)],
        specification=spec, priority_date=priority)


@pytest.fixture
def target() -> Patent:
    return make_patent(
        "US8123456B2", "Sensor driven zone irrigation system",
        "1. An irrigation system comprising wireless soil moisture sensor nodes "
        "and a controller scheduling watering of zones.",
        "Wireless soil moisture sensor nodes feed a zone irrigation controller.",
        date(2008, 4, 10))


@pytest.fixture
def store() -> BM25Store:
    bm25 = BM25Store()
    bm25.index([
        make_patent(
            "US7000001B1", "Wireless soil moisture sensor network",
            "1. A soil moisture sensor node transmitting readings over a wireless "
            "mesh network to an irrigation controller.",
            "Sensor nodes measure soil moisture and relay readings over a wireless "
            "mesh network to an irrigation controller scheduling watering of zones.",
            date(2001, 3, 1)),
        make_patent(
            "US7000002B1", "Soil moisture probe for irrigation control",
            "1. A capacitive soil moisture probe providing readings to an "
            "irrigation controller for watering control.",
            "A capacitive soil moisture probe drives an irrigation controller.",
            date(2002, 6, 15)),
    ])
    return bm25


@pytest.fixture
def seeded_graph(tmp_path) -> ConceptGraph:
    graph = ConceptGraph(tmp_path / "graph")
    graph.add(ConceptNode(
        canonical_name="WIRELESS_SOIL_MOISTURE_SENSING",
        aliases=["wireless soil moisture sensor nodes",
                 "soil humidity radio sensors",
                 "wireless moisture probe network"],
        stage=STAGE_REVIEWED_CANONICAL))
    return graph


def test_guided_injects_concept_aliases_and_harvests_pairs(
        target: Patent, store: BM25Store, seeded_graph: ConceptGraph,
        tmp_path) -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "search_patents", "arguments": {
            "keywords": ["soil", "moisture", "sensor", "wireless"]}}]},
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [{"number": "US7000001B1", "why": "mesh network nodes",
                            "confidence": 0.8,
                            "key_passages": ["sensor nodes relay readings"]}],
            "rationale": "one strong angle"}}]},
    ])
    match_pairs = MatchPairStore(tmp_path / "graph")
    guided = GuidedSearch(keyword_store=store, llm=llm,
                          concept_graph=seeded_graph, match_pair_store=match_pairs)

    limitation = "wireless soil moisture sensor nodes"
    session = guided.start_with_patent("invalidity", target, claims=[1],
                                       key_limitations=[limitation])
    session = guided.execute(session)

    # (a) the concept-graph aliases were injected into the agent conversation
    # via the same persisted-user-message channel as key_limitations
    texts = [block.get("text", "")
             for message in llm.tool_conversations[0]
             for block in message["content"] if isinstance(block, dict)]
    angle_messages = [t for t in texts if "ADDITIONAL QUERY ANGLES" in t]
    assert len(angle_messages) == 1
    assert "soil humidity radio sensors" in angle_messages[0]
    assert "wireless moisture probe network" in angle_messages[0]
    assert limitation not in angle_messages[0]  # the input phrasing is not echoed
    assert any("Key claim limitations" in t and limitation in t for t in texts)

    # (b) the round's matches were harvested automatically, unreviewed
    harvested = match_pairs.filter(search_id=session.id)
    assert harvested, "execute must harvest match pairs into the configured store"
    assert all(p.outcome == "unreviewed" for p in harvested)
    assert {p.patent_id for p in harvested} == {"US7000001B1"}
    assert harvested[0].query_limitation == limitation
    assert harvested[0].matched_text == "sensor nodes relay readings"
    assert harvested[0].created_at == date.today().isoformat()

    # (c) result feedback naming the patent flips its pairs' outcomes
    session = guided.apply_result_feedback(session, SearchFeedback(
        results=[ResultFeedback(patent_number="US7000001B1", relevant=True)]))
    accepted = match_pairs.filter(patent_id="US7000001B1", search_id=session.id)
    assert accepted and all(p.outcome == "accepted" for p in accepted)
    assert all(p.feedback_type == "teaches_limitation" for p in accepted)


def test_guided_without_graph_behaves_as_before(target: Patent,
                                                store: BM25Store) -> None:
    """No concept_graph / match_pair_store configured -> no injection, no
    harvesting, degraded keys-free mode untouched."""
    guided = GuidedSearch(keyword_store=store, llm=None)
    session = guided.start_with_patent("invalidity", target, claims=[1])
    session = guided.execute(session)
    assert session.state == "awaiting_result_feedback"
    assert session.last_results
    assert session.params["stop_reason"] == "degraded"


def test_guided_degraded_mode_harvests_with_claim_fallback(
        target: Patent, store: BM25Store, tmp_path) -> None:
    """Without key_limitations the harvest falls back to the claims' text."""
    match_pairs = MatchPairStore(tmp_path / "graph")
    guided = GuidedSearch(keyword_store=store, llm=None,
                          match_pair_store=match_pairs)
    session = guided.start_with_patent("invalidity", target, claims=[1])
    session = guided.execute(session)
    harvested = match_pairs.filter(search_id=session.id)
    assert harvested
    assert all(p.outcome == "unreviewed" for p in harvested)
    assert all("irrigation system" in p.query_limitation for p in harvested)
