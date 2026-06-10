"""Tests for the PatentToolset, TOOL_SPECS, dispatch, and the OpenAI layer."""

from __future__ import annotations

import json
from datetime import date

import pytest

from patentkit.integrations.openai_tools import handle_tool_call, openai_tool_definitions
from patentkit.integrations.toolset import TOOL_SPECS, PatentToolset, dispatch
from patentkit.models import Citation, Claim, Patent, PatentNumber
from patentkit.search.bm25 import BM25Store


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
def toolset(tmp_path) -> PatentToolset:
    store = BM25Store()
    store.index([
        make_patent(
            "US8123456B2", "Sensor driven zone irrigation system",
            "1. An irrigation system comprising wireless soil moisture sensor nodes "
            "and a controller scheduling watering of zones.",
            "Wireless soil moisture sensor nodes feed a zone irrigation controller.",
            date(2008, 4, 10),
            citations=[Citation(patent_number=PatentNumber.parse("US7000003B1"),
                                is_examiner=True)],
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
    return PatentToolset(keyword_store=store, session_dir=str(tmp_path / "sessions"))


# --------------------------------------------------------------- TOOL_SPECS

def test_tool_specs_schema_sanity() -> None:
    assert TOOL_SPECS, "TOOL_SPECS must not be empty"
    names = [spec["name"] for spec in TOOL_SPECS]
    assert len(names) == len(set(names)), "tool names must be unique"
    for spec in TOOL_SPECS:
        assert spec["name"] and isinstance(spec["name"], str)
        assert spec["description"] and isinstance(spec["description"], str)
        params = spec["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params["required"], list)
        for required in params["required"]:
            assert required in params["properties"]
    json.dumps(TOOL_SPECS)  # specs themselves are JSON-serializable


def test_tool_specs_cover_every_toolset_method() -> None:
    for spec in TOOL_SPECS:
        method = getattr(PatentToolset, spec["name"], None)
        assert callable(method), f"PatentToolset.{spec['name']} missing"


def test_search_patents_spec_enumerates_full_query_params() -> None:
    spec = next(s for s in TOOL_SPECS if s["name"] == "search_patents")
    properties = spec["parameters"]["properties"]
    for param in ("keywords", "required_keywords", "excluded_keywords", "text",
                  "minimum_match", "fields", "art_classes", "inventors", "assignees",
                  "before_date", "after_date", "countries", "exclude_numbers", "limit"):
        assert param in properties, f"search_patents schema missing {param}"


# ----------------------------------------------------------------- dispatch

def test_dispatch_unknown_tool_returns_error(toolset: PatentToolset) -> None:
    out = dispatch(toolset, "no_such_tool", {})
    assert "error" in out and "no_such_tool" in out["error"]


def test_dispatch_bad_arguments_returns_error(toolset: PatentToolset) -> None:
    out = dispatch(toolset, "get_patent", {"bogus_argument": 1})
    assert "error" in out


def test_search_patents_through_dispatch(toolset: PatentToolset) -> None:
    out = dispatch(toolset, "search_patents", {
        "keywords": ["soil", "moisture", "irrigation"],
        "before_date": "2008-04-10",
        "exclude_numbers": ["US7000002B1"],
        "limit": 10,
    })
    json.dumps(out)  # JSON-serializable
    assert out["count"] >= 1
    numbers = [r["patent_number"] for r in out["results"]]
    assert "US7000001B1" in numbers
    assert "US7000002B1" not in numbers          # explicit exclusion
    assert "US8123456B2" not in numbers          # priority on the cutoff date
    assert all(r["passages"] for r in out["results"])


def test_get_and_index_through_dispatch(toolset: PatentToolset, tmp_path) -> None:
    got = dispatch(toolset, "get_patent", {"number": "US7000001B1"})
    assert got["title"] == "Wireless soil moisture sensor network"

    extra = make_patent("US7000009B1", "Drip emitter", "1. A drip emitter.",
                        "A drip irrigation emitter.", date(1999, 1, 1))
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(extra.model_dump_json() + "\n")
    out = dispatch(toolset, "index_patents", {"jsonl_path": str(jsonl)})
    assert out["indexed"] == 1 and not out["errors"]
    assert dispatch(toolset, "get_patent", {"number": "US7000009B1"})["title"] == "Drip emitter"


def test_estimate_search_time_through_dispatch(toolset: PatentToolset) -> None:
    small = dispatch(toolset, "estimate_search_time", {"n_queries": 2})
    big = dispatch(toolset, "estimate_search_time", {"n_queries": 2, "corpus_size": 10_000_000,
                                                     "charting_claims": 3})
    assert small["seconds"] > 0 and big["seconds"] > small["seconds"]
    assert "human" in small


def test_guided_flow_through_dispatch(toolset: PatentToolset) -> None:
    started = dispatch(toolset, "guided_search_start", {
        "search_type": "invalidity", "patent_number": "US8123456B2", "claims": [1],
    })
    assert "error" not in started
    session_id = started["session_id"]
    assert started["state"] == "awaiting_plan_feedback"
    assert started["plan"]["queries"]
    assert started["estimated_seconds"] > 0

    revised = dispatch(toolset, "guided_search_feedback", {
        "session_id": session_id,
        "feedback": {"queries": [{"query_index": 0, "verdict": "too_narrow"}],
                     "free_text": "more synonyms please"},
    })
    assert revised["state"] == "awaiting_plan_feedback"

    executed = dispatch(toolset, "guided_search_execute", {"session_id": session_id})
    assert executed["state"] == "awaiting_result_feedback"
    numbers = [r["patent_number"] for r in executed["results"]]
    assert numbers and "US7000003B1" not in numbers  # examiner-art exclusion default
    json.dumps(executed)

    status = dispatch(toolset, "guided_search_status", {"session_id": session_id})
    assert status["state"] == "awaiting_result_feedback"
    assert status["n_results"] == len(numbers)
    assert status["feedback_rounds"] == 1

    # result feedback queues another iteration
    again = dispatch(toolset, "guided_search_feedback", {
        "session_id": session_id,
        "feedback": {"results": [{"patent_number": numbers[0], "relevant": False}]},
    })
    assert again["state"] == "searching" and again["iteration"] == 1
    rerun = dispatch(toolset, "guided_search_execute", {"session_id": session_id})
    assert numbers[0] not in [r["patent_number"] for r in rerun["results"]]


def test_guided_status_unknown_session(toolset: PatentToolset) -> None:
    out = dispatch(toolset, "guided_search_status", {"session_id": "nope"})
    assert "error" in out


def test_cluster_and_eval_degrade_gracefully(toolset: PatentToolset) -> None:
    clusters = dispatch(toolset, "cluster_patents", {"numbers": ["US7000001B1"]})
    evals = dispatch(toolset, "run_eval", {})
    # Either the optional module is installed and returns data, or we get a
    # helpful error dict — never an exception.
    assert isinstance(clusters, dict) and isinstance(evals, dict)
    if "error" in clusters:
        assert "viz" in clusters["error"]


def test_notify_without_notifiers(toolset: PatentToolset) -> None:
    out = dispatch(toolset, "notify", {"subject": "s", "body": "b"})
    assert out["sent"] == 0


# ------------------------------------------------------------- OpenAI layer

def test_openai_tool_definitions_shape() -> None:
    definitions = openai_tool_definitions()
    assert len(definitions) == len(TOOL_SPECS)
    for definition in definitions:
        assert definition["type"] == "function"
        function = definition["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert function["parameters"]["type"] == "object"
    json.dumps(definitions)


def test_handle_tool_call_returns_json_string(toolset: PatentToolset) -> None:
    raw = handle_tool_call(toolset, "search_patents",
                           json.dumps({"keywords": ["soil"], "limit": 5}))
    parsed = json.loads(raw)
    assert parsed["count"] >= 1

    bad = json.loads(handle_tool_call(toolset, "search_patents", "{not json"))
    assert "error" in bad
