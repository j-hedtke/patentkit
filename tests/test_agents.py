"""Tests for the agents layer: planner, invalidity pipeline, guided loop."""

from __future__ import annotations

import json
from datetime import date

import pytest

from patentkit.agents import (
    GuidedSearch,
    GuidedSearchSession,
    InvaliditySearchAgent,
    QueryFeedback,
    ResultFeedback,
    SearchFeedback,
    SearchPlan,
    estimate_search_seconds,
    humanize_seconds,
    plan_search,
)
from patentkit.agents.guided import SessionStore
from patentkit.agents.planner import PlannedQuery, QuerySpec, _fallback_keywords
from patentkit.models import Citation, Claim, Patent, PatentNumber
from patentkit.search.bm25 import BM25Store
from tests.fakes import FakeLLM


# ----------------------------------------------------------------- fixtures

def make_patent(number: str, title: str, claim: str, spec: str, priority: date,
                **kwargs) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(number),
        title=title,
        abstract=spec[:150],
        claims=[Claim(number=1, text=claim)],
        specification=spec,
        priority_date=priority,
        **kwargs,
    )


@pytest.fixture
def target() -> Patent:
    return make_patent(
        "US8123456B2", "Sensor driven zone irrigation system",
        "1. An irrigation system comprising wireless soil moisture sensor nodes "
        "and a controller scheduling watering of zones from transmitted moisture readings.",
        "The system combines wireless soil moisture sensor nodes with a zone "
        "irrigation controller scheduling watering from transmitted readings.",
        date(2008, 4, 10),
        citations=[Citation(patent_number=PatentNumber.parse("US7000003B1"), is_examiner=True)],
        family=[PatentNumber.parse("US8123457A1")],
    )


@pytest.fixture
def corpus() -> list[Patent]:
    """Six synthetic patents: 2 relevant, 1 examiner-cited, 1 family,
    1 post-priority, 1 weakly related."""
    return [
        make_patent(
            "US7000001B1", "Wireless soil moisture sensor network",
            "1. A soil moisture sensor node transmitting moisture readings over a "
            "wireless mesh network to an irrigation controller.",
            "Sensor nodes measure soil moisture and relay readings over a wireless "
            "mesh network to an irrigation controller that schedules watering of zones. "
            "Wireless soil moisture sensor irrigation controller zones watering.",
            date(2001, 3, 1),
        ),
        make_patent(
            "US7000002B1", "Soil moisture probe for irrigation control",
            "1. A capacitive soil moisture probe providing readings to an irrigation "
            "controller for watering control.",
            "A capacitive soil moisture probe is calibrated and its moisture readings "
            "drive an irrigation controller for watering.",
            date(2002, 6, 15),
        ),
        make_patent(  # examiner-cited -> excluded
            "US7000003B1", "Irrigation valve scheduling with moisture sensors",
            "1. An irrigation controller actuating valves using soil moisture sensor "
            "readings transmitted wirelessly.",
            "Soil moisture sensor readings transmitted wirelessly drive irrigation "
            "valve scheduling for watering zones.",
            date(2003, 1, 20),
        ),
        make_patent(  # family member -> excluded (pre-cutoff so date isn't the reason)
            "US8123457A1", "Sensor driven zone irrigation system (continuation)",
            "1. An irrigation system comprising wireless soil moisture sensor nodes "
            "and a controller scheduling watering of zones.",
            "Wireless soil moisture sensor nodes feed a zone irrigation controller "
            "scheduling watering from transmitted moisture readings.",
            date(2005, 1, 1),
        ),
        make_patent(  # post-priority -> excluded by date cutoff
            "US9000001B2", "Machine learning irrigation scheduling",
            "1. Training a model on wireless soil moisture sensor readings to predict "
            "irrigation watering demand for zones.",
            "A model trained on wireless soil moisture sensor readings predicts zone "
            "irrigation watering demand.",
            date(2014, 5, 1),
        ),
        make_patent(
            "US7000004B1", "Greenhouse climate telemetry",
            "1. A greenhouse telemetry system reporting temperature and humidity "
            "readings over a radio link.",
            "Greenhouse temperature and humidity sensor readings are logged over a "
            "radio link; soil moisture and irrigation are mentioned only in passing.",
            date(2000, 9, 5),
        ),
    ]


@pytest.fixture
def store(corpus: list[Patent]) -> BM25Store:
    bm25 = BM25Store()
    bm25.index(corpus)
    return bm25


# ------------------------------------------------------------------ planner

def test_plan_search_falls_back_on_bad_llm_json(target: Patent) -> None:
    llm = FakeLLM(responses=["this is definitely {{{ not json"])
    plan = plan_search("invalidity", patent=target, llm=llm, n_queries=4)
    assert plan.search_type == "invalidity"
    assert plan.target == "US8123456B2"
    assert len(plan.queries) == 1  # single fallback query
    keywords = plan.queries[0].query.keywords
    assert keywords, "fallback keywords must not be empty"
    assert "irrigation" in keywords  # distinctive title term
    # invalidity plans carry the prior-art cutoff and examiner exclusions
    assert plan.queries[0].query.before_date == date(2008, 4, 10)
    assert "US7000003B1" in plan.exclusions
    assert "fallback" in plan.rationale.lower() or "Heuristic" in plan.rationale


def test_plan_search_without_llm_uses_fallback(target: Patent) -> None:
    plan = plan_search("invalidity", patent=target, llm=None)
    assert len(plan.queries) == 1
    assert plan.estimated_seconds and plan.estimated_seconds > 0


def test_fallback_keywords_skip_boilerplate(target: Patent) -> None:
    keywords = _fallback_keywords(target)
    for boilerplate in ("comprising", "the", "system", "a"):
        assert boilerplate not in keywords


def test_query_spec_round_trip() -> None:
    spec = QuerySpec(keywords=["soil"], exclude_numbers=["US7000003B1"],
                     before_date=date(2008, 4, 10), limit=7)
    query = spec.to_search_query()
    assert query.keywords == ["soil"]
    assert query.exclude_numbers == [PatentNumber.parse("US7000003B1")]
    assert query.before_date == date(2008, 4, 10)
    back = QuerySpec.from_search_query(query)
    assert back.exclude_numbers == ["US7000003B1"]
    assert back.limit == 7


def _plan(n_queries: int) -> SearchPlan:
    return SearchPlan(search_type="invalidity", target="t", rationale="r",
                      queries=[PlannedQuery(purpose="q", query=QuerySpec())
                               for _ in range(n_queries)])


def test_estimate_search_seconds_monotonicity() -> None:
    base = estimate_search_seconds(_plan(2), corpus_size=1000, with_llm_rerank=False)
    assert base > 0
    # more queries -> longer
    assert estimate_search_seconds(_plan(4), 1000, False) > base
    # bigger corpus -> longer
    assert estimate_search_seconds(_plan(2), 1_000_000, False) > base
    # LLM rerank -> longer
    assert estimate_search_seconds(_plan(2), 1000, True) > base
    # charting -> longer, monotone in claim count
    one = estimate_search_seconds(_plan(2), 1000, False, charting_claims=1)
    two = estimate_search_seconds(_plan(2), 1000, False, charting_claims=2)
    assert base < one < two


def test_humanize_seconds() -> None:
    assert humanize_seconds(45) == "45 seconds"
    assert "minutes" in humanize_seconds(300)
    assert "hours" in humanize_seconds(7200)


# ------------------------------------------------------- invalidity pipeline

def test_invalidity_search_end_to_end_keys_free(target: Patent, store: BM25Store) -> None:
    agent = InvaliditySearchAgent(keyword_store=store, llm=None)
    messages: list[str] = []
    result = agent.search(target, claims=[1], final_k=10, progress=messages.append)

    numbers = [r["patent_number"] for r in result.results]
    assert numbers, "expected at least one prior-art result"
    # examiner-cited, family, self, and post-cutoff art are all excluded
    assert "US7000003B1" not in numbers          # examiner-cited
    assert "US8123457A1" not in numbers          # family member
    assert "US8123456B2" not in numbers          # the target itself
    assert "US9000001B2" not in numbers          # priority 2014 > 2008 cutoff
    assert "US7000003B1" in result.excluded["examiner_cited"]
    assert "US8123457A1" in result.excluded["family"]
    # the genuinely relevant references made it through
    assert "US7000001B1" in numbers
    # every result carries highlighted passages
    for ranked in result.results:
        assert ranked["passages"], f"{ranked['patent_number']} has no passages"
        assert all(p["text"] for p in ranked["passages"])
    # date cutoff recorded in the executed params; LLM stage was skipped
    assert result.plan_or_params["before_date"] == "2008-04-10"
    assert result.plan_or_params["llm_rerank"] is False
    assert result.timing["total"] >= 0
    assert any("stage 1" in m for m in messages)


def test_invalidity_examiner_exclusion_can_be_disabled(target: Patent, store: BM25Store) -> None:
    agent = InvaliditySearchAgent(keyword_store=store, llm=None)
    result = agent.search(target, claims=[1], exclude_examiner_art=False, final_k=10)
    numbers = [r["patent_number"] for r in result.results]
    assert "US7000003B1" in numbers
    assert "examiner_cited" not in result.excluded


def test_invalidity_stage3_llm_rerank_changes_order(target: Patent, store: BM25Store) -> None:
    # Without an LLM, US7000001B1 outranks US7000002B1 on keywords alone.
    keys_free = InvaliditySearchAgent(keyword_store=store, llm=None)
    baseline = [r["patent_number"] for r in keys_free.search(target, claims=[1]).results]
    assert baseline.index("US7000001B1") < baseline.index("US7000002B1")

    # FakeLLM: first call returns keywords, second the stage-3 relevance scores.
    llm = FakeLLM(responses=[
        ["soil", "moisture", "sensor", "irrigation", "wireless"],
        [
            {"number": "US7000002B1", "score": 10, "why": "discloses every limitation"},
            {"number": "US7000001B1", "score": 2, "why": "missing the controller"},
        ],
    ])
    agent = InvaliditySearchAgent(keyword_store=store, llm=llm)
    result = agent.search(target, claims=[1])
    numbers = [r["patent_number"] for r in result.results]
    assert numbers.index("US7000002B1") < numbers.index("US7000001B1"), \
        "LLM stage-3 scores should reorder the ranking"
    top = result.results[numbers.index("US7000002B1")]
    assert top["why"] == "discloses every limitation"
    assert result.plan_or_params["llm_rerank"] is True


def test_file_wrapper_enrichment_failures_are_skipped(target: Patent, store: BM25Store) -> None:
    class ExplodingWrapper:
        def enrich_patent(self, patent):
            raise RuntimeError("boom")

    agent = InvaliditySearchAgent(keyword_store=store, llm=None, file_wrapper=ExplodingWrapper())
    result = agent.search(target, claims=[1])
    assert result.results  # pipeline survives enrichment failure


# ----------------------------------------------------------------- guided

def test_guided_session_full_state_machine_round_trip(target: Patent, store: BM25Store,
                                                      tmp_path) -> None:
    sessions = SessionStore(tmp_path / "sessions")
    guided = GuidedSearch(keyword_store=store, llm=None, session_store=sessions)

    # start -> planning done, awaiting plan feedback, estimate attached
    session = guided.start_with_patent("invalidity", target, claims=[1])
    assert session.state == "awaiting_plan_feedback"
    assert session.plan is not None and session.plan.queries
    assert session.params["estimated_seconds"] > 0
    assert "second" in session.params["estimated_human"] or "minute" in session.params["estimated_human"]

    # JSON round-trip survives with full fidelity
    payload = session.model_dump_json()
    restored = GuidedSearchSession.model_validate_json(payload)
    assert restored == session
    # and the persisted copy can be reloaded by a fresh store
    fresh = SessionStore(tmp_path / "sessions")
    assert fresh.get(session.id) is not None

    # plan feedback revises the plan in place
    session = guided.apply_plan_feedback(restored, SearchFeedback(
        queries=[QueryFeedback(query_index=0, verdict="too_narrow", note="add synonyms")],
        free_text="cover mesh networking angles too",
    ))
    assert session.state == "awaiting_plan_feedback"
    assert len(session.feedback_history) == 1

    # execute -> results land on the session
    session = guided.execute(session)
    assert session.state == "awaiting_result_feedback"
    assert session.last_results
    first_numbers = [r["patent_number"] for r in session.last_results]
    assert "US7000003B1" not in first_numbers  # examiner-art exclusion holds

    # result feedback: mark one irrelevant -> excluded next iteration
    victim = first_numbers[0]
    session = guided.apply_result_feedback(session, SearchFeedback(
        results=[ResultFeedback(patent_number=victim, relevant=False)],
    ))
    assert session.state == "searching"
    assert session.iteration == 1
    assert victim in session.plan.exclusions

    # serialize mid-loop and resume in a "new process"
    resumed = GuidedSearchSession.model_validate_json(session.model_dump_json())
    session = guided.execute(resumed)
    assert session.state == "awaiting_result_feedback"
    assert victim not in [r["patent_number"] for r in session.last_results]

    session = guided.finish(session)
    assert session.state == "done"
    with pytest.raises(ValueError):
        guided.execute(session)


def test_guided_start_requires_resolvable_patent(store: BM25Store) -> None:
    guided = GuidedSearch(keyword_store=store, llm=None)

    def fetch(number: str):  # offline resolver — no network in tests
        raise ValueError(f"patent {number} not found")

    with pytest.raises(ValueError):
        guided.start("invalidity", target_patent_number="US9999999B9", fetch=fetch)


def test_guided_fto_flow(store: BM25Store) -> None:
    guided = GuidedSearch(keyword_store=store, llm=None)
    session = guided.start("fto", product_description="wireless soil moisture sensor "
                                                      "for irrigation watering of garden zones")
    session = guided.execute(session)
    assert session.state == "awaiting_result_feedback"
    assert json.dumps(session.model_dump(mode="json"))  # fully serializable
