"""Tests for the analysis skills using FakeLLM (no network, no extras)."""

from patentkit.analysis import (
    build_claim_chart,
    check_antecedent_basis,
    generate_keywords,
    split_atomic_limitations,
)
from patentkit.models import Claim, Patent, PatentNumber
from patentkit.parsing import parse_claims
from tests.fakes import FakeLLM


def make_patent(claims: list[Claim]) -> Patent:
    return Patent(
        patent_number=PatentNumber.parse("US10123456B2"),
        title="Widget assembly",
        abstract="A widget with a frame and a motor.",
        claims=claims,
    )


CLAIM_TEXT = "A widget comprising: a frame; and a motor coupled to the frame."


class TestSplitAtomicLimitations:
    def test_split_with_valid_and_invalid_spans(self):
        claim = Claim(number=1, text=CLAIM_TEXT)
        lim_a = "A widget comprising a frame"
        lim_b = "a motor coupled to the frame"
        llm = FakeLLM(
            responses=[
                [lim_a, lim_b],
                [
                    {"limitation": lim_a, "start": 0, "end": 27},
                    {"limitation": lim_b, "start": 50, "end": 10},  # invalid: start > end
                ],
            ]
        )
        limitations = split_atomic_limitations(claim, llm=llm)
        assert [lim.text for lim in limitations] == [lim_a, lim_b]
        # valid span kept from the LLM
        assert limitations[0].span == (0, 27)
        # invalid span fell back to case-insensitive substring search
        start = CLAIM_TEXT.lower().find(lim_b.lower())
        assert limitations[1].span == (start, start + len(lim_b))
        assert len(llm.prompts) == 2

    def test_unfindable_limitation_gets_none_span(self):
        claim = Claim(number=1, text=CLAIM_TEXT)
        llm = FakeLLM(responses=[["a paraphrased requirement not in the claim"], []])
        limitations = split_atomic_limitations(claim, llm=llm)
        assert limitations[0].span is None


class TestBuildClaimChart:
    def test_end_to_end_and_coverage_math(self):
        patent = make_patent([Claim(number=1, text="A device comprising: lim A; and lim B.")])
        llm = FakeLLM(
            responses=[
                ["lim A", "lim B"],  # split
                [],                  # span mapping (falls back to substring search)
                # reference US111: lim A disclosed, lim B not
                {"status": "disclosed", "reasoning": "teaches A", "quotes": ["the art shows A"]},
                {"status": "not disclosed", "reasoning": "silent on B", "quotes": []},
                # reference US222: lim A not, lim B disclosed
                {"status": "not_disclosed", "reasoning": "silent on A", "quotes": []},
                {"status": "disclosed", "reasoning": "teaches B", "quotes": ["the art shows B"]},
            ]
        )
        chart = build_claim_chart(
            patent, 1,
            references=[("US111", "reference one text"), ("US222", "reference two text")],
            llm=llm,
        )
        assert chart.query_patent == "US10123456B2"
        assert chart.claim_number == 1
        assert [lim.text for lim in chart.limitations] == ["lim A", "lim B"]
        assert len(chart.references) == 2
        assert chart.references[0].findings[0].status == "disclosed"
        assert chart.references[0].findings[0].quotes == ["the art shows A"]
        assert chart.references[0].findings[1].status == "not_disclosed"

        # coverage math: each reference discloses 1 of 2 limitations
        assert chart.coverage_summary() == {"US111": 0.5, "US222": 0.5}
        # but together they cover both
        assert chart.combined_coverage() == 1.0
        # 2 split calls + 4 assess calls
        assert len(llm.prompts) == 6

    def test_locator_attaches_citations(self):
        patent = make_patent([Claim(number=1, text="A device comprising: lim A.")])
        llm = FakeLLM(
            responses=[
                ["lim A"],
                [],
                {"status": "disclosed", "reasoning": "r", "quotes": ["quoted passage"]},
            ]
        )
        chart = build_claim_chart(
            patent, 1, references=[("US111", "ref text")],
            llm=llm, locator=lambda passage: "col. 3, ll. 45-52",
        )
        assert chart.references[0].findings[0].citation == "col. 3, ll. 45-52"

    def test_missing_claim_raises(self):
        patent = make_patent([Claim(number=1, text="A device.")])
        try:
            build_claim_chart(patent, 9, references=[], llm=FakeLLM())
        except ValueError as exc:
            assert "9" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestAntecedentBasis:
    def test_clean_claim_chain_has_no_issues(self):
        claims = parse_claims(
            "1. A beverage container comprising:\n"
            "a body defining an interior chamber;\n"
            "a lid removably coupled to the body, the lid including a vent; and\n"
            "wherein the vent is configured to release steam from the interior chamber.\n"
            "2. The beverage container of claim 1, wherein the body comprises stainless steel.\n"
        )
        assert check_antecedent_basis(claims) == []

    def test_missing_antecedent_flagged(self):
        claims = parse_claims(
            "1. A widget comprising:\na frame; and\nwherein the gear ratio exceeds two.\n"
        )
        issues = check_antecedent_basis(claims)
        assert len(issues) == 1
        assert "Claim 1" in issues[0]
        assert "gear ratio" in issues[0]

    def test_dependent_claim_inherits_parent_antecedents(self):
        claims = parse_claims(
            "1. A machine comprising a rotor.\n"
            "2. The machine of claim 1, wherein the rotor is balanced.\n"
        )
        assert check_antecedent_basis(claims) == []

    def test_said_reference_without_antecedent_flagged(self):
        claims = parse_claims("1. A pump comprising a housing, said impeller being sealed.")
        issues = check_antecedent_basis(claims)
        assert any("impeller" in issue for issue in issues)


class TestGenerateKeywords:
    def test_voting_ranks_by_frequency(self):
        patent = make_patent([Claim(number=1, text=CLAIM_TEXT)])
        llm = FakeLLM(
            responses=[
                ["alpha", "beta", "gamma"],
                ["beta", "gamma"],
                ["Beta", "delta"],  # case-insensitive vote for beta
            ]
        )
        keywords = generate_keywords(patent, llm=llm, votes=3)
        assert keywords[0] == "beta"  # 3 votes
        assert keywords[1] == "gamma"  # 2 votes
        assert set(keywords) == {"alpha", "beta", "gamma", "delta"}
        assert len(llm.prompts) == 3

    def test_caps_at_top_n(self):
        patent = make_patent([Claim(number=1, text=CLAIM_TEXT)])
        many = [f"kw{i}" for i in range(30)]
        llm = FakeLLM(responses=[many, many, many])
        keywords = generate_keywords(patent, llm=llm, votes=3)
        assert len(keywords) == 15
        assert keywords == [f"kw{i}" for i in range(15)]  # first-seen tie-break
