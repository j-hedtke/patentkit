"""Tests for the chat-UX tool upgrades: markdown claim charts, single-
limitation charting, chart caching for DOCX export, key-limitation
summaries, trace narratives, and per-round execute budgets.

Everything runs offline with FakeLLM — no network, no SDKs.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from patentkit.integrations.toolset import PatentToolset, dispatch
from patentkit.models import Claim, Patent, PatentNumber
from patentkit.search.bm25 import BM25Store
from tests.fakes import FakeLLM

# Claim of the target patent; limitations are PRECOMPUTED structural units
# (deterministic split — preamble at "comprising", elements at ";"), so the
# segments below are exactly what Claim.get_limitations() yields. No LLM
# call is ever needed for splitting.
CLAIM = ("1. An irrigation system comprising: wireless soil moisture sensor "
         "nodes; and a controller scheduling watering of zones.")
LIM_PREAMBLE = "1. An irrigation system comprising:"
LIM_SENSORS = "wireless soil moisture sensor nodes; and"
LIM_CONTROLLER = "a controller scheduling watering of zones."
LIMITATIONS = [LIM_PREAMBLE, LIM_SENSORS, LIM_CONTROLLER]
LABELS = ["1[pre]", "1[a]", "1[b]"]


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


def make_store() -> BM25Store:
    store = BM25Store()
    store.index([
        make_patent(
            "US8123456B2", "Sensor driven zone irrigation system", CLAIM,
            "Wireless soil moisture sensor nodes feed a zone irrigation controller.",
            date(2008, 4, 10),
        ),
        make_patent(
            "US7000001B1", "Wireless soil moisture sensor network",
            "1. A soil moisture sensor node transmitting moisture readings over a "
            "wireless mesh network to an irrigation controller.",
            "Sensor nodes measure soil moisture and relay readings wirelessly to an "
            "irrigation controller scheduling watering.",
            date(2001, 3, 1),
        ),
        make_patent(
            "US7000002B1", "Soil moisture probe for irrigation control",
            "1. A capacitive soil moisture probe providing readings to an irrigation "
            "controller for watering control.",
            "A capacitive soil moisture probe drives an irrigation controller.",
            date(2002, 6, 15),
        ),
    ])
    return store


def make_toolset(tmp_path, llm) -> PatentToolset:
    return PatentToolset(keyword_store=make_store(), llm=llm,
                         session_dir=str(tmp_path / "sessions"))


def assess(status: str, reasoning: str, quotes: list[str]) -> dict:
    return {"status": status, "reasoning": reasoning, "quotes": quotes}


def chart_llm(extra_responses: list | None = None) -> FakeLLM:
    """FakeLLM scripted for one build_claim_chart against US7000001B1.

    Splitting is deterministic (no LLM call), so only the three per-
    limitation disclosure assessments are scripted.
    """
    return FakeLLM(responses=[
        assess("disclosed", "irrigation system taught", ["an irrigation controller"]),
        assess("disclosed", "mesh sensor nodes taught", ["sensor node transmitting readings"]),
        assess("partial", "scheduling only implied", ["controller scheduling watering"]),
        *(extra_responses or []),
    ])


# -------------------------------------------------------- build_claim_chart

def test_build_claim_chart_returns_markdown_and_structured_fields(tmp_path) -> None:
    toolset = make_toolset(tmp_path, chart_llm())
    out = dispatch(toolset, "build_claim_chart", {
        "patent_number": "US8123456B2", "claim_number": 1,
        "reference_numbers": ["US7000001B1"],
    })
    assert "error" not in out
    json.dumps(out)
    # structured fields preserved for programmatic use
    assert [lim["text"] for lim in out["limitations"]] == LIMITATIONS
    assert [lim["label"] for lim in out["limitations"]] == LABELS
    assert out["coverage_summary"]["US7000001B1"] == pytest.approx(2 / 3)
    # ready-to-display markdown: bold bracket label + verbatim text
    md = out["markdown"]
    assert md.startswith("## Claim Chart — US8123456B2, Claim 1")
    assert f"**1[b]** {LIM_CONTROLLER}" in md
    assert "US7000001B1 (Wireless soil moisture sensor network)" in md
    assert "**Disclosed**" in md and "**Partial**" in md
    assert "“controller scheduling watering”" in md


def test_build_claim_chart_limitations_filter_restricts_markdown(tmp_path) -> None:
    toolset = make_toolset(tmp_path, chart_llm())
    out = dispatch(toolset, "build_claim_chart", {
        "patent_number": "US8123456B2", "claim_number": 1,
        "reference_numbers": ["US7000001B1"],
        "limitations_filter": ["controller scheduling"],
    })
    assert "error" not in out
    # full chart kept in the structured fields ...
    assert len(out["limitations"]) == 3
    # ... but the markdown is restricted to the matching row
    md = out["markdown"]
    assert "Filtered to 1 of 3 limitation(s)" in md
    assert LIM_CONTROLLER in md
    assert LIM_SENSORS not in md
    assert LIM_PREAMBLE not in md


def test_build_claim_chart_filter_with_no_match_falls_back_to_full(tmp_path) -> None:
    toolset = make_toolset(tmp_path, chart_llm())
    out = dispatch(toolset, "build_claim_chart", {
        "patent_number": "US8123456B2", "claim_number": 1,
        "reference_numbers": ["US7000001B1"],
        "limitations_filter": ["quantum flux capacitor"],
    })
    assert "matched no" in out["markdown"]
    assert LIM_SENSORS in out["markdown"]  # full chart shown


# --------------------------------------------------------- chart_limitation

def test_chart_limitation_reuses_cached_assessments(tmp_path) -> None:
    llm = chart_llm(extra_responses=[
        assess("not_disclosed", "no scheduling in the probe art", []),
    ])
    toolset = make_toolset(tmp_path, llm)
    dispatch(toolset, "build_claim_chart", {
        "patent_number": "US8123456B2", "claim_number": 1,
        "reference_numbers": ["US7000001B1"],
    })
    calls_after_chart = len(llm.prompts)  # 3 assess (split is deterministic)

    # same limitation + same reference: fully served from the cache
    out = dispatch(toolset, "chart_limitation", {
        "limitation": "controller scheduling",
        "patent": "US8123456B2", "claim_number": 1,
        "references": ["US7000001B1"],
    })
    assert "error" not in out
    assert len(llm.prompts) == calls_after_chart  # NO new LLM calls
    assert out["reused_assessments"] == ["US7000001B1"]
    assert out["new_assessments"] == []
    assert out["limitation"] == LIM_CONTROLLER
    assert out["label"] == "1[b]"
    md = out["markdown"]
    assert md.startswith("## Limitation Chart — US8123456B2, Claim 1")
    assert LIM_CONTROLLER in md
    assert "| Reference | Status | Reasoning | Quotes |" in md
    assert "**Partial**" in md

    # new reference: exactly one new assessment, merged into the cached chart
    out2 = dispatch(toolset, "chart_limitation", {
        "limitation": "controller scheduling",
        "patent": "US8123456B2", "claim_number": 1,
        "references": ["US7000001B1", "US7000002B1"],
    })
    assert len(llm.prompts) == calls_after_chart + 1
    assert out2["reused_assessments"] == ["US7000001B1"]
    assert out2["new_assessments"] == ["US7000002B1"]
    assert "US7000002B1" in out2["markdown"]
    assert "**Not disclosed**" in out2["markdown"]


def test_chart_limitation_without_prior_chart(tmp_path) -> None:
    llm = FakeLLM(responses=[
        # no split response needed — limitations are precomputed
        assess("disclosed", "scheduling controller taught", ["controller scheduling watering"]),
    ])
    toolset = make_toolset(tmp_path, llm)
    out = dispatch(toolset, "chart_limitation", {
        "limitation": LIM_CONTROLLER,  # verbatim form also accepted
        "patent": "US8123456B2", "claim_number": 1,
        "references": ["US7000001B1"],
    })
    assert "error" not in out
    assert out["new_assessments"] == ["US7000001B1"]
    assert out["markdown"].startswith("## Limitation Chart")
    assert LIM_CONTROLLER in out["markdown"]
    # the single-limitation chart is now cached for export
    assert toolset._cached_chart("US8123456B2", 1) is not None


def test_chart_limitation_unknown_limitation_errors_helpfully(tmp_path) -> None:
    toolset = make_toolset(tmp_path, FakeLLM())  # no LLM calls expected at all
    out = dispatch(toolset, "chart_limitation", {
        "limitation": "a quantum flux capacitor",
        "patent": "US8123456B2", "claim_number": 1,
        "references": ["US7000001B1"],
    })
    assert "error" in out
    assert LIM_CONTROLLER in out["error"]  # lists the actual limitations


# --------------------------------------------------- export_claim_chart_docx

def test_export_without_cached_chart_errors(tmp_path) -> None:
    toolset = make_toolset(tmp_path, FakeLLM())
    out = dispatch(toolset, "export_claim_chart_docx", {
        "patent": "US8123456B2", "claim_number": 1,
    })
    assert "error" in out
    assert "build_claim_chart" in out["error"]


def test_export_uses_cached_chart_without_llm_calls(tmp_path) -> None:
    pytest.importorskip("docx")
    llm = chart_llm()
    toolset = make_toolset(tmp_path, llm)
    dispatch(toolset, "build_claim_chart", {
        "patent_number": "US8123456B2", "claim_number": 1,
        "reference_numbers": ["US7000001B1"],
    })
    calls = len(llm.prompts)

    out = dispatch(toolset, "export_claim_chart_docx", {
        "patent": "US8123456B2", "claim_number": 1,
    })
    assert "error" not in out
    assert len(llm.prompts) == calls  # export never re-runs LLM calls
    path = Path(out["path"])
    assert path.is_absolute() and path.suffix == ".docx" and path.exists()
    # default location: exports/ under the session dir
    assert path.parent == (tmp_path / "sessions" / "exports").resolve()
    assert out["references"] == ["US7000001B1"]
    assert out["limitations"] == 3

    # kind-code-agnostic lookup + explicit out_path
    explicit = tmp_path / "out" / "my_chart.docx"
    out2 = dispatch(toolset, "export_claim_chart_docx", {
        "patent": "US8123456", "claim_number": 1, "out_path": str(explicit),
    })
    assert Path(out2["path"]) == explicit.resolve()
    assert explicit.exists()


# ----------------------------------------------------- summarize_key_limitations

def test_summarize_key_limitations_degrades_without_odp_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("USPTO_ODP_API_KEY", raising=False)
    toolset = PatentToolset(keyword_store=make_store(),
                            session_dir=str(tmp_path / "sessions"))  # keys-free, no LLM
    out = dispatch(toolset, "summarize_key_limitations", {
        "patent": "US8123456B2", "claim_number": 1,
    })
    assert out["mode"] == "claim_split_only"
    assert "USPTO_ODP_API_KEY" in out["note"]
    # the PRECOMPUTED limitations are returned — no LLM needed at all
    assert out["key_limitations"] == LIMITATIONS
    assert [lim["label"] for lim in out["limitations"]] == LABELS
    md = out["markdown"]
    assert md.startswith("## Claim limitations — US8123456B2, claim 1")
    assert "USPTO_ODP_API_KEY" in md
    assert f"- **1[b]** {LIM_CONTROLLER}" in md
    json.dumps(out)


def test_summarize_key_limitations_uses_enriched_file_wrapper(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("USPTO_ODP_API_KEY", raising=False)
    store = make_store()
    enriched = make_patent(
        "US9999999B2", "Enriched irrigation patent", CLAIM,
        "spec text", date(2010, 1, 1),
        file_wrapper_text=("=== CTNF 2009-01-01 ===\nRejected over Smith.\n"
                           "=== REM 2009-06-01 ===\nApplicant amended claim 1 to add "
                           "a controller scheduling watering of zones."),
    )
    store.index([enriched])
    llm = FakeLLM(responses=[
        # only the wrapper-summarization call — splitting is deterministic
        {"key_limitations": [{"limitation": LIM_CONTROLLER,
                              "why": "added by amendment to overcome Smith"}],
         "summary": "Allowance followed the controller-scheduling amendment."},
    ])
    toolset = PatentToolset(keyword_store=store, llm=llm,
                            session_dir=str(tmp_path / "sessions"))
    out = dispatch(toolset, "summarize_key_limitations", {
        "patent": "US9999999B2", "claim_number": 1,
    })
    assert out["mode"] == "file_wrapper"
    assert out["key_limitations"] == [LIM_CONTROLLER]
    assert out["details"][0]["why"] == "added by amendment to overcome Smith"
    md = out["markdown"]
    assert md.startswith("## Key limitations — US9999999B2, claim 1")
    assert LIM_CONTROLLER in md and "overcome Smith" in md
    # the prosecution history reached the LLM prompt
    assert "Applicant amended claim 1" in llm.prompts[-1]


def test_summarize_key_limitations_fetches_wrapper_with_key(tmp_path, monkeypatch) -> None:
    import patentkit.connectors.inference.file_wrapper as fw

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def app_number_for_patent(self, patent_number):
            return "16123456"

        def get_file_wrapper_text(self, app_number):
            assert app_number == "16123456"
            return "=== REM === argued the controller scheduling limitation"

    monkeypatch.setenv("USPTO_ODP_API_KEY", "test-key")
    monkeypatch.setattr(fw, "FileWrapperClient", StubClient)
    llm = FakeLLM(responses=[
        {"key_limitations": [{"limitation": "controller scheduling watering",
                              "why": "argued for allowance"}],
         "summary": "Argued, not amended."},
    ])
    toolset = make_toolset(tmp_path, llm)
    out = dispatch(toolset, "summarize_key_limitations", {
        "patent": "US8123456B2", "claim_number": 1,
    })
    assert out["mode"] == "file_wrapper"
    # the LLM's loose text was snapped to the verbatim limitation
    assert out["key_limitations"] == [LIM_CONTROLLER]
    assert "argued the controller scheduling limitation" in llm.prompts[-1]


# ------------------------------------------- guided search: key limitations

def test_guided_start_injects_key_limitations_into_agent_context(tmp_path) -> None:
    llm = FakeLLM(tool_script=[
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [], "rationale": "nothing found"}}]},
    ])
    toolset = make_toolset(tmp_path, llm)
    started = dispatch(toolset, "guided_search_start", {
        "search_type": "invalidity", "patent_number": "US8123456B2", "claims": [1],
        "key_limitations": ["a controller scheduling watering of zones"],
    })
    assert "error" not in started
    dispatch(toolset, "guided_search_execute", {"session_id": started["session_id"]})
    first_conversation = json.dumps(llm.tool_conversations[0])
    assert "Key claim limitations to prioritize" in first_conversation
    assert "a controller scheduling watering of zones" in first_conversation


# ------------------------------- guided execute: budgets, resume, progress

def test_execute_budget_overrides_and_resume_same_conversation(tmp_path) -> None:
    llm = FakeLLM(tool_script=[
        # round 1: one search, then the step budget (max_steps=1) is breached
        {"text": "Starting broad.", "tool_calls": [{"name": "search_patents", "arguments": {
            "keywords": ["soil", "moisture", "wireless"]}}]},
        # grace round after the wrap-up message
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [{"number": "US7000001B1", "why": "mesh sensors",
                            "confidence": 0.7}],
            "rationale": "budget hit"}}]},
        # second execute (resumed conversation)
        {"tool_calls": [{"name": "finish", "arguments": {
            "candidates": [{"number": "US7000001B1", "why": "mesh sensors",
                            "confidence": 0.9}],
            "rationale": "confirmed"}}]},
    ])
    toolset = make_toolset(tmp_path, llm)
    started = dispatch(toolset, "guided_search_start", {
        "search_type": "invalidity", "patent_number": "US8123456B2", "claims": [1]})
    session_id = started["session_id"]

    executed = dispatch(toolset, "guided_search_execute", {
        "session_id": session_id, "max_steps": 1, "budget_seconds": 30})
    assert executed["stop_reason"] == "max_steps"  # the override took effect
    trace = dispatch(toolset, "get_search_trace", {"session_id": session_id})
    assert "max_steps=1" in trace["markdown"]
    assert "budget_seconds=30" in trace["markdown"]

    # progress markdown: round number, queries, shortlist, steering hint
    progress = executed["progress"]
    assert "round 1 complete" in progress
    assert "`search_patents`" in progress
    assert "US7000001B1" in progress
    assert "guided_search_feedback" in progress and "guided_search_execute" in progress

    # second execute resumes the SAME conversation
    n_runs_first = len(llm.tool_conversations)
    rerun = dispatch(toolset, "guided_search_execute", {"session_id": session_id})
    assert rerun["stop_reason"] == "finish_tool"
    assert [r["patent_number"] for r in rerun["results"]] == ["US7000001B1"]
    resumed = llm.tool_conversations[n_runs_first]
    resumed_text = json.dumps(resumed)
    assert "Find prior art that invalidates" in resumed_text  # original task kept
    assert "budget hit" in resumed_text                       # round-1 finish kept
    assert len(resumed) > len(llm.tool_conversations[0])
    assert "round 2 complete" in rerun["progress"]


def test_degraded_execute_progress_is_labeled(tmp_path) -> None:
    toolset = PatentToolset(keyword_store=make_store(),
                            session_dir=str(tmp_path / "sessions"))  # no LLM
    started = dispatch(toolset, "guided_search_start", {
        "search_type": "invalidity", "patent_number": "US8123456B2", "claims": [1]})
    executed = dispatch(toolset, "guided_search_execute", {
        "session_id": started["session_id"]})
    assert executed["stop_reason"] == "degraded"
    assert "DEGRADED" in executed["progress"]
    assert "not agentic-mode performance" in executed["progress"]


# ------------------------------------------------------- trace narrative

def test_search_trace_markdown_is_round_grouped_narrative(tmp_path) -> None:
    llm = FakeLLM(tool_script=[
        {"text": "I will start with mesh-network terminology.",
         "tool_calls": [{"name": "search_patents", "arguments": {
             "keywords": ["soil", "moisture", "wireless"]}}]},
        {"text": "US7000001B1 looks on point; shortlisting it.",
         "tool_calls": [{"name": "shortlist", "arguments": {"candidates": [
             {"number": "US7000001B1", "why": "mesh sensor network"}]}},
             {"name": "finish", "arguments": {
                 "candidates": [{"number": "US7000001B1", "why": "mesh sensor network",
                                 "confidence": 0.8}],
                 "rationale": "done"}}]},
    ])
    toolset = make_toolset(tmp_path, llm)
    started = dispatch(toolset, "guided_search_start", {
        "search_type": "invalidity", "patent_number": "US8123456B2", "claims": [1]})
    session_id = started["session_id"]
    dispatch(toolset, "guided_search_execute", {"session_id": session_id})

    trace = dispatch(toolset, "get_search_trace", {"session_id": session_id})
    md = trace["markdown"]
    assert md.startswith("# Reasoning trace — agentic invalidity search")
    # one section per round, with the agent's thinking text
    assert "## Round 1" in md and "## Round 2" in md
    assert "I will start with mesh-network terminology." in md
    # queries as inline code, paired with result counts
    assert '`search_patents {"keywords": ["soil", "moisture", "wireless"]}`' in md
    assert "result(s)" in md
    # shortlist + stop reason + evolution
    assert "candidate(s) accepted" in md
    assert "`finish_tool`" in md
    assert "## Shortlist evolution" in md and "US7000001B1" in md

    # feedback injected on resume shows up in the next run's trace
    dispatch(toolset, "guided_search_feedback", {
        "session_id": session_id,
        "feedback": {"free_text": "focus on mesh networking"},
    })
    llm._tool_script.append({"tool_calls": [{"name": "finish", "arguments": {
        "candidates": [{"number": "US7000001B1", "why": "still best", "confidence": 0.9}],
        "rationale": "refined"}}]})
    dispatch(toolset, "guided_search_execute", {"session_id": session_id})
    trace2 = dispatch(toolset, "get_search_trace", {"session_id": session_id})
    assert "## Injected user feedback" in trace2["markdown"]
    assert "focus on mesh networking" in trace2["markdown"]


def test_search_patents_resolves_patent_number_queries_by_lookup(tmp_path) -> None:
    """Patent-number-shaped query terms never match BM25 text fields; the tool
    must answer with a direct number lookup + guidance instead of noise."""
    from patentkit.integrations.toolset import _looks_like_patent_number

    assert _looks_like_patent_number("US10491679B2")
    assert _looks_like_patent_number("US 2006/0235700 A1")
    assert not _looks_like_patent_number("voice command interface")

    ts = make_toolset(tmp_path, FakeLLM())
    indexed = str(ts.keyword_store.all_patents()[0].patent_number)
    out = ts.search_patents(keywords=[indexed])
    assert out["results"] == [] and out["lookups"][0]["indexed"] is True
    missing = ts.search_patents(text="US99999999B9")
    assert missing["lookups"][0]["indexed"] is False
    assert "get_patent" in missing["note"]
