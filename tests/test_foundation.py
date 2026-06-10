"""Tests for the foundation layer: models, config, BM25, vector store, routing."""

from datetime import date

import pytest

from patentkit.config import MissingKeyError, Keyring, resolve_key
from patentkit.llm.routing import ReasoningEffort, choose_model
from patentkit.models import (
    Assignee,
    Citation,
    Claim,
    Classification,
    Inventor,
    Patent,
    PatentNumber,
)
from patentkit.search import (
    BM25Store,
    HashingEmbeddings,
    InMemoryVectorStore,
    SearchQuery,
)


def make_patent(num: str, *, title="", spec="", claims=(), filing=None, cpc=(),
                inventors=(), assignees=()) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse(num),
        title=title,
        specification=spec,
        claims=[Claim(number=i + 1, text=t) for i, t in enumerate(claims)],
        filing_date=filing,
        classifications=[Classification(code=c) for c in cpc],
        inventors=[Inventor(name=n) for n in inventors],
        assignees=[Assignee(name=n) for n in assignees],
    )


class TestPatentNumber:
    @pytest.mark.parametrize("raw,country,number,kind", [
        ("US10123456B2", "US", "10123456", "B2"),
        ("10,123,456", "US", "10123456", None),
        ("EP1234567A1", "EP", "1234567", "A1"),
        ("US 2020/0123456 A1", "US", "20200123456", "A1"),
        ("us9000000", "US", "9000000", None),
    ])
    def test_parse(self, raw, country, number, kind):
        pn = PatentNumber.parse(raw)
        assert (pn.country_code, pn.number, pn.kind_code) == (country, number, kind)

    def test_equivalent_ignores_kind(self):
        assert PatentNumber.parse("US7654321B2").equivalent(PatentNumber.parse("US7654321"))
        assert not PatentNumber.parse("US7654321").equivalent(PatentNumber.parse("EP7654321"))

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            PatentNumber.parse("not a patent")


class TestMerge:
    def test_fidelity_and_citation_flags(self):
        from patentkit.models import SourceRecord
        low = Patent(
            patent_number=PatentNumber.parse("US1000000"),
            title="low fidelity title", abstract="only-low abstract",
            citations=[Citation(patent_number=PatentNumber.parse("US900000"), is_examiner=True)],
            sources=[SourceRecord(source="serpapi", fidelity=1)],
        )
        high = Patent(
            patent_number=PatentNumber.parse("US1000000B2"),
            title="high fidelity title",
            citations=[Citation(patent_number=PatentNumber.parse("US900000"), is_applicant=True)],
            sources=[SourceRecord(source="google_patents", fidelity=3)],
        )
        merged = low.merge(high)
        assert merged.title == "high fidelity title"
        assert merged.abstract == "only-low abstract"  # filled from secondary
        assert len(merged.citations) == 1
        assert merged.citations[0].is_examiner and merged.citations[0].is_applicant
        assert {s.source for s in merged.sources} == {"serpapi", "google_patents"}

    def test_merge_different_patents_raises(self):
        a = make_patent("US1"); b = make_patent("US2")
        with pytest.raises(ValueError):
            a.merge(b)


class TestConfig:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert resolve_key("ANTHROPIC_API_KEY", "explicit") == "explicit"
        assert resolve_key("ANTHROPIC_API_KEY") == "from-env"

    def test_missing_raises_with_pointer(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(MissingKeyError, match="VOYAGE_API_KEY"):
            resolve_key("VOYAGE_API_KEY")

    def test_keyring(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        ring = Keyring(slack_webhook_url="https://hooks.slack.example/x")
        assert ring.get("SLACK_WEBHOOK_URL") == "https://hooks.slack.example/x"


class TestRouting:
    def test_defaults(self):
        assert choose_model("high").model == "claude-fable-5"
        assert choose_model(ReasoningEffort.LOW).model == "claude-haiku-4-5"
        openai_high = choose_model("high", provider="openai")
        assert openai_high.model.startswith("gpt-5") and openai_high.reasoning_effort == "high"

    def test_unknown_provider(self):
        with pytest.raises(ValueError):
            choose_model("low", provider="nope")


class TestBM25:
    def setup_method(self):
        self.store = BM25Store()
        self.store.index([
            make_patent("US1", title="wireless charging coil",
                        spec="a resonant coil transfers wireless power inductively " * 20,
                        claims=("A coil for wireless power transfer.",),
                        filing=date(2015, 1, 1), cpc=("H02J50",), assignees=("Acme Power",)),
            make_patent("US2", title="solar panel mount",
                        spec="a bracket mounts photovoltaic panels to roofs " * 20,
                        filing=date(2018, 6, 1), cpc=("H02S20",), inventors=("Ada Lovelace",)),
            make_patent("US3", title="wireless network router",
                        spec="a router routes wireless network packets " * 20,
                        filing=date(2020, 3, 1), cpc=("H04L45",)),
        ])

    def test_ranking_and_passages(self):
        results = self.store.search(SearchQuery(keywords=["wireless", "coil", "power"]))
        assert results[0].patent_number.number == "1"
        assert results[0].passages and "coil" in results[0].passages[0].text.lower()

    def test_date_cutoff(self):
        results = self.store.search(SearchQuery(keywords=["wireless"], before_date=date(2016, 1, 1),
                                                minimum_match=1))
        assert [r.patent_number.number for r in results] == ["1"]

    def test_art_class_filter(self):
        results = self.store.search(SearchQuery(keywords=["wireless"], art_classes=["H04L"],
                                                minimum_match=1))
        assert [r.patent_number.number for r in results] == ["3"]

    def test_excluded_keywords(self):
        results = self.store.search(SearchQuery(keywords=["wireless"], excluded_keywords=["router"],
                                                minimum_match=1))
        assert all(r.patent_number.number != "3" for r in results)

    def test_exclude_numbers(self):
        results = self.store.search(SearchQuery(
            keywords=["wireless"], minimum_match=1,
            exclude_numbers=[PatentNumber.parse("US1")],
        ))
        assert all(r.patent_number.number != "1" for r in results)

    def test_inventor_and_assignee_filters(self):
        assert [r.patent_number.number for r in self.store.search(
            SearchQuery(keywords=["panel"], inventors=["lovelace"], minimum_match=1))] == ["2"]
        assert [r.patent_number.number for r in self.store.search(
            SearchQuery(keywords=["coil"], assignees=["acme"], minimum_match=1))] == ["1"]

    def test_minimum_match(self):
        # require all 3 keywords -> only US1 has coil+power+wireless
        results = self.store.search(SearchQuery(keywords=["wireless", "coil", "power"], minimum_match=3))
        assert [r.patent_number.number for r in results] == ["1"]


class TestVectorStore:
    def test_roundtrip(self):
        store = InMemoryVectorStore(HashingEmbeddings(dimensions=128))
        store.index([
            make_patent("US1", title="wireless charging", spec="coil inductive power " * 50),
            make_patent("US2", title="solar mount", spec="photovoltaic bracket roof " * 50),
        ])
        results = store.search_text("inductive coil charging")
        assert results[0].patent_number.number == "1"
        assert results[0].passages
