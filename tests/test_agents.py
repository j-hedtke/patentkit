"""Tests for the agents layer: planner preview, agentic search, guided loop."""

from __future__ import annotations

import json
from datetime import date

import pytest

from patentkit.agents import (
    AgenticSearchRunner,
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
from patentkit.agents.fto_search import FtoSearchAgent
from patentkit.agents.guided import SessionStore
from patentkit.agents.infringement_search import InfringementSearchAgent
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

def test_plan_search_is_deterministic_preview_no_llm_call(target: Patent) -> None:
    llm = FakeLLM()
    plan = plan_search("invalidity", patent=target, llm=llm)
    assert llm.prompts == [], "plan preview must not call the LLM"
    assert plan.search_type == "invalidity"
    assert plan.target == "US8123456B2"
    assert plan.queries, "preview must have at least one starting angle"
    keywords = plan.queries[0].query.keywords
    assert "irrigation" in keywords  # distinctive title term
    # invalidity plans carry the prior-art cutoff and examiner exclusions
    assert all(q.query.before_date == date(2008, 4, 10) for q in plan.queries)
    assert "US7000003B1" in plan.exclusions
    assert plan.estimated_seconds and plan.estimated_seconds > 0
    assert "agent" in plan.rationale


def test_plan_search_without_llm_works(target: Patent) -> None:
    plan = plan_search("invalidity", patent=target, llm=None)
    assert plan.queries
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


def test_estimate_search_seconds_monotonicity_agentic_model() -> None:
    base = estimate_search_seconds(_plan(2), corpus_size=1000, with_llm_rerank=False)
    assert base > 0
    # more angles -> longer
    assert estimate_search_seconds(_plan(4), 1000, False) > base
    # bigger corpus -> longer
    assert estimate_search_seconds(_plan(2), 1_000_000, False) > base
    # agentic (LLM-driven) run -> longer than the degraded single pass
    agentic = estimate_search_seconds(_plan(2), 1000, True)
    assert agentic > base
    assert estimate_search_seconds(_plan(4), 1000, True) > agentic
    # default-shaped searches must fit the <=3-minute budget envelope
    assert estimate_search_seconds(_plan(3), 100_000, True) <= 180
    # charting -> longer, monotone in claim count
    one = estimate_search_seconds(_plan(2), 1000, False, charting_claims=1)
    two = estimate_search_seconds(_plan(2), 1000, False, charting_claims=2)
    assert base < one < two


def test_humanize_seconds() -> None:
    assert humanize_seconds(45) == "45 seconds"
    assert "minutes" in humanize_seconds(300)
    assert "hours" in humanize_seconds(7200)


# ------------------------------------------------ invalidity: degraded mode

def test_invalidity_degraded_keys_free(target: Patent, store: BM25Store) -> None:
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
    # the genuinely relevant references made it through, sorted best-first
    assert "US7000001B1" in numbers
    scores = [r["score"] for r in result.results]
    assert scores == sorted(scores, reverse=True)
    for ranked in result.results:
        assert ranked["passages"], f"{ranked['patent_number']} has no passages"
    # degraded mode is clearly labeled
    assert result.plan_or_params["mode"] == "degraded_keyword_only"
    assert result.plan_or_params["before_date"] == "2008-04-10"
    assert result.stop_reason == "degraded"
    assert result.trace is None
    assert result.timing["total"] >= 0
    assert any("degraded" in m for m in messages)


def test_invalidity_examiner_exclusion_can_be_disabled(target: Patent, store: BM25Store) -> None:
    agent = InvaliditySearchAgent(keyword_store=store, llm=None)
    result = agent.search(target, claims=[1], exclude_examiner_art=False, final_k=10)
    numbers = [r["patent_number"] for r in result.results]
    assert "US7000003B1" in numbers
    assert "examiner_cited" not in result.excluded


def test_file_wrapper_enrichment_failures_are_skipped(target: Patent, store: BM25Store) -> None:
    class ExplodingWrapper:
        def enrich_patent(self, patent):
            raise RuntimeError("boom")

    agent = InvaliditySearchAgent(keyword_store=store, llm=None, file_wrapper=ExplodingWrapper())
    result = agent.search(target, claims=[1])
    assert result.results  # search survives enrichment failure


def test_file_wrapper_enrichment_adds_examiner_exclusions(target: Patent,
                                                          store: BM25Store) -> None:
    class EnrichingWrapper:
        def enrich_patent(self, patent):
            enriched = patent.model_copy(deep=True)
            enriched.citations.append(Citation(
                patent_number=PatentNumber.parse("US7000002B1"), is_examiner=True))
            return enriched

    agent = InvaliditySearchAgent(keyword_store=store, llm=None,
                                  file_wrapper=EnrichingWrapper())
    result = agent.search(target, claims=[1], final_k=10)
    assert "US7000002B1" in result.excluded["examiner_cited"]
    assert "US7000002B1" not in [r["patent_number"] for r in result.results]


# ------------------------------------------------- invalidity: agentic mode

def agentic_script() -> list[dict]:
    """Two search angles, a shortlist (with one bogus entry), then finish
    (which also tries to sneak in an excluded number)."""
    return [
        {"text": "Trying the sensor-network angle first.",
         "tool_calls": [{"name": "search_patents", "arguments": {
             "keywords": ["soil", "moisture", "sensor", "wireless"], "limit": 10}}]},
        # different keywords; targets the examiner-cited patent and tries to
        # loosen the cutoff — both must be neutralized by the tool layer
        {"tool_calls": [{"name": "search_patents", "arguments": {
            "keywords": ["irrigation", "valve", "scheduling", "machine", "learning"],
            "before_date": "2030-01-01", "limit": 10}}]},
        {"tool_calls": [{"name": "shortlist", "arguments": {"candidates": [
            {"number": "US7000001B1", "why": "mesh network nodes feed a controller"},
            {"number": "US9999999B9", "why": "never appeared in any search"},
        ]}}]},
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [
                {"number": "US7000002B1", "why": "probe drives the controller",
                 "confidence": 0.9, "key_passages": ["capacitive soil moisture probe"]},
                {"number": "US7000001B1", "why": "wireless mesh sensor network",
                 "confidence": 0.7},
                {"number": "US7000003B1", "why": "examiner art (must be rejected)",
                 "confidence": 0.99},
            ],
            "rationale": "both angles converged",
            "suggested_next_queries": ["capacitive probe calibration"],
        }}]},
    ]


def test_agentic_invalidity_end_to_end(target: Patent, store: BM25Store) -> None:
    llm = FakeLLM(tool_script=agentic_script())
    agent = InvaliditySearchAgent(keyword_store=store, llm=llm)
    steps = []
    result = agent.search(target, claims=[1], final_k=50, on_step=steps.append)

    # ranked exactly as scripted (excluded sneak-in rejected), best first
    numbers = [r["patent_number"] for r in result.results]
    assert numbers == ["US7000002B1", "US7000001B1"]
    assert result.results[0]["score"] == 0.9
    assert result.results[1]["score"] == 0.7
    # titles hydrated from the store; key passages preserved
    assert result.results[0]["title"] == "Soil moisture probe for irrigation control"
    assert result.results[0]["passages"][0]["text"] == "capacitive soil moisture probe"
    assert result.results[0]["why"] == "probe drives the controller"
    assert result.stop_reason == "finish_tool"
    assert result.plan_or_params["mode"] == "agentic"
    assert result.plan_or_params["suggested_next_queries"] == ["capacitive probe calibration"]
    assert result.conversation, "agentic result must carry the resumable conversation"
    assert steps, "on_step must receive trace steps"

    # excluded numbers NEVER returned by the search tool, even when asked for
    trace = result.trace
    assert trace is not None
    search_results = [s for s in trace.steps
                      if s.kind == "tool_result" and s.tool_name == "search_patents"]
    assert len(search_results) == 2
    for step in search_results:
        assert "US7000003B1" not in step.content   # examiner-cited
        assert "US8123457A1" not in step.content   # family
        assert "US8123456B2" not in step.content   # self
        assert "US9000001B2" not in step.content   # post-cutoff (clamp held)
    # the bogus shortlist entry was rejected; the valid one kept
    assert trace.shortlist_history == [[{"number": "US7000001B1",
                                         "why": "mesh network nodes feed a controller"}]]

    # the markdown trace contains the queries the agent issued
    markdown = trace.to_markdown()
    assert "search_patents" in markdown
    assert '"soil"' in markdown and '"wireless"' in markdown
    assert '"valve"' in markdown and '"scheduling"' in markdown
    assert "Queries issued" in markdown
    json.dumps(trace.model_dump(mode="json"))  # trace is fully serializable


def test_agentic_invalidity_truncated_run_falls_back_to_shortlist(
        target: Patent, store: BM25Store) -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "search_patents", "arguments": {
            "keywords": ["soil", "moisture", "sensor"]}}]},
        {"tool_calls": [{"name": "shortlist", "arguments": {"candidates": [
            {"number": "US7000001B1", "why": "good", "key_passage": "mesh network"}]}}]},
        {"tool_calls": [{"name": "search_patents", "arguments": {"keywords": ["probe"]}}]},
        {"tool_calls": [{"name": "search_patents", "arguments": {"keywords": ["valve"]}}]},
    ])
    agent = InvaliditySearchAgent(keyword_store=store, llm=llm)
    result = agent.search(target, claims=[1], max_steps=3)
    assert result.stop_reason == "max_steps"
    numbers = [r["patent_number"] for r in result.results]
    assert numbers == ["US7000001B1"]  # the working shortlist
    assert "shortlist" in result.plan_or_params["rationale"]


def test_agentic_runner_requires_llm(store: BM25Store) -> None:
    runner = AgenticSearchRunner(store, llm=None)
    with pytest.raises(ValueError, match="requires an llm"):
        runner.run("invalidity", exclusions={})


def test_agentic_runner_rejects_unknown_search_type(store: BM25Store) -> None:
    runner = AgenticSearchRunner(store, llm=FakeLLM(tool_script=[]))
    with pytest.raises(ValueError, match="search_type"):
        runner.run("bogus", exclusions={})


# -------------------------------------------------------------- fto / infr.

def test_fto_degraded_keys_free(store: BM25Store) -> None:
    agent = FtoSearchAgent(keyword_store=store, llm=None)
    result = agent.search("wireless soil moisture sensor for irrigation watering "
                          "of garden zones", in_force_only=False, final_k=5)
    assert result.results
    assert result.plan_or_params["mode"] == "degraded_keyword_only"
    assert result.stop_reason == "degraded"
    assert result.requires_attorney_review is True


def test_infringement_degraded_token_overlap(target: Patent) -> None:
    agent = InfringementSearchAgent(llm=None)
    result = agent.search(target, claims=[1], product_candidates=[
        {"name": "SmartSprinkler", "description": "wireless soil moisture sensor "
                                                  "nodes controlling watering of zones"},
        {"name": "Toaster", "description": "browns bread"},
    ])
    assert result.results[0]["name"] == "SmartSprinkler"
    assert result.results[0]["score"] > result.results[1]["score"]
    assert "degraded" in result.results[0]["rationale"]
    assert result.stop_reason == "degraded"


def test_infringement_agentic_ranks_products(target: Patent) -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [
                {"number": "SmartSprinkler", "why": "practices every limitation",
                 "confidence": 0.85},
                {"number": "Toaster", "why": "no irrigation function", "confidence": 0.05},
            ],
            "rationale": "evidence review complete",
        }}]},
    ])
    agent = InfringementSearchAgent(llm=llm)
    result = agent.search(target, claims=[1], product_candidates=[
        {"name": "SmartSprinkler", "description": "wireless watering controller",
         "url": "https://example.com/ss"},
        {"name": "Toaster", "description": "browns bread"},
    ], evidence_texts=["datasheet: soil moisture nodes"])
    assert [r["name"] for r in result.results] == ["SmartSprinkler", "Toaster"]
    assert result.results[0]["url"] == "https://example.com/ss"
    assert result.results[0]["score"] == 0.85
    assert result.stop_reason == "finish_tool"
    assert result.trace is not None


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

    # plan feedback adjusts the preview and seeds the agent's guidance
    session = guided.apply_plan_feedback(restored, SearchFeedback(
        queries=[QueryFeedback(query_index=0, verdict="too_narrow", note="add synonyms")],
        free_text="cover mesh networking angles too",
    ))
    assert session.state == "awaiting_plan_feedback"
    assert len(session.feedback_history) == 1
    assert any("mesh networking" in g for g in session.params["pre_run_guidance"])

    # execute -> results land on the session
    session = guided.execute(session)
    assert session.state == "awaiting_result_feedback"
    assert session.last_results
    assert session.params["stop_reason"] == "degraded"
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
    assert session.params["pending_feedback"], "feedback is queued for the agent"

    # serialize mid-loop and resume in a "new process"
    resumed = GuidedSearchSession.model_validate_json(session.model_dump_json())
    session = guided.execute(resumed)
    assert session.state == "awaiting_result_feedback"
    assert victim not in [r["patent_number"] for r in session.last_results]

    session = guided.finish(session)
    assert session.state == "done"
    with pytest.raises(ValueError):
        guided.execute(session)


def test_guided_agentic_feedback_resumes_same_conversation(target: Patent,
                                                           store: BM25Store) -> None:
    llm = FakeLLM(tool_script=[
        # --- first execution ---
        {"tool_calls": [{"name": "search_patents", "arguments": {
            "keywords": ["soil", "moisture", "sensor"]}}]},
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [{"number": "US7000001B1", "why": "mesh network",
                            "confidence": 0.8}],
            "rationale": "first pass"}}]},
        # --- resumed execution after feedback ---
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [{"number": "US7000002B1", "why": "capacitive probe",
                            "confidence": 0.9},
                           {"number": "US7000001B1", "why": "mesh network",
                            "confidence": 0.6}],
            "rationale": "re-ranked per feedback"}}]},
    ])
    guided = GuidedSearch(keyword_store=store, llm=llm)
    session = guided.start_with_patent("invalidity", target, claims=[1])
    session = guided.execute(session)
    assert [r["patent_number"] for r in session.last_results] == ["US7000001B1"]
    assert session.params["stop_reason"] == "finish_tool"
    assert session.params["conversation"], "conversation persisted for resumption"
    first_convo_len = len(session.params["conversation"])
    trace_summary_queries = session.params["trace"]["queries"]
    assert trace_summary_queries and trace_summary_queries[0]["tool"] == "search_patents"

    session = guided.apply_result_feedback(session, SearchFeedback(
        results=[ResultFeedback(patent_number="US7000001B1", relevant=True,
                                note="good but rank probes higher")],
        free_text="prefer capacitive probe art",
    ))
    session = guided.execute(session)

    # the second run RESUMED the same conversation (no fresh task message) ...
    resumed_convo = llm.tool_conversations[-1]
    assert len(resumed_convo) > first_convo_len - 1
    texts = [b.get("text", "") for m in resumed_convo
             for b in m["content"] if isinstance(b, dict)]
    # ... the original task message is still its first message ...
    assert any("Find prior art that invalidates US8123456B2" in t for t in texts)
    # ... and the injected feedback message is present
    assert any("USER FEEDBACK" in t and "capacitive probe art" in t for t in texts)
    # results updated from the resumed run; US7000002B1 was seen in run 1's
    # search results, so the reseeded validation accepts it
    assert [r["patent_number"] for r in session.last_results] == \
        ["US7000002B1", "US7000001B1"]
    # the resumed run's trace records the feedback event
    assert any("capacitive probe art" in f for f in session.params["trace"]["feedback"])


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
