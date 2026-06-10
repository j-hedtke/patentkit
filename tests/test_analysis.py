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

# Realistic multi-element claim, modeled on US5946647A claim 1.
DETECT_CLAIM_TEXT = (
    "A computer-based system for detecting structures in data and performing "
    "actions on detected structures, comprising: an input device for receiving "
    "data; an output device for presenting the data; a memory storing "
    "information including program routines including an analyzer server for "
    "detecting structures in the data, and for linking actions to the detected "
    "structures; and a processing unit coupled to the input device, the output "
    "device, and the memory for controlling the execution of the program "
    "routines."
)

DETECT_PREAMBLE = (
    "A computer-based system for detecting structures in data and performing "
    "actions on detected structures, comprising:"
)
DETECT_ELEMENT_1 = "an input device for receiving data;"


def normalize(text: str) -> str:
    return " ".join(text.split())


def assert_verbatim_in_order(limitations, claim_text):
    """Every limitation is a verbatim substring (after whitespace
    normalization) with a span, and spans strictly increase."""
    norm_claim = normalize(claim_text)
    assert limitations, "expected at least one limitation"
    for lim in limitations:
        assert normalize(lim.text) in norm_claim
        assert lim.span is not None
        start, end = lim.span
        assert claim_text[start:end] == lim.text
    starts = [lim.span[0] for lim in limitations]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)  # strictly increasing


class TestSplitAtomicLimitations:
    def test_verbatim_segments_sorted_into_claim_order(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        # LLM returns verbatim segments but OUT of order and with sloppy
        # internal whitespace; code must locate, re-slice, and sort them.
        llm = FakeLLM(
            responses=[
                [
                    "an  output device for presenting the data;",
                    DETECT_PREAMBLE,
                    DETECT_ELEMENT_1,
                ]
            ]
        )
        limitations = split_atomic_limitations(claim, llm=llm)
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        # (c) preamble first, then the first element after the preamble
        assert limitations[0].text == DETECT_PREAMBLE
        assert limitations[1].text == DETECT_ELEMENT_1
        # whitespace-sloppy segment was re-sliced verbatim from the claim
        assert limitations[2].text == "an output device for presenting the data;"
        assert len(llm.prompts) == 1  # single split call, no span-mapping call

    def test_llm_none_uses_deterministic_structural_split(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        limitations = split_atomic_limitations(claim, llm=None)
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        texts = [lim.text for lim in limitations]
        assert texts[0] == DETECT_PREAMBLE
        assert texts[1] == DETECT_ELEMENT_1
        assert texts[2] == "an output device for presenting the data;"
        # "; and" delimiter stays with the preceding element
        assert texts[3].endswith("to the detected structures; and")
        assert texts[4].startswith("a processing unit coupled to")
        # together the segments cover essentially the whole claim
        assert normalize(" ".join(texts)) == normalize(DETECT_CLAIM_TEXT)

    def test_paraphrasing_llm_falls_back_to_verbatim_segments(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        llm = FakeLLM(
            responses=[
                [
                    "The system comprises an input device for receiving data.",
                    "There is an output device.",
                ]
            ]
        )
        limitations = split_atomic_limitations(claim, llm=llm)
        # paraphrases never survive: result is verbatim, in order
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        assert limitations[0].text == DETECT_PREAMBLE
        assert limitations[1].text == DETECT_ELEMENT_1
        for lim in limitations:
            assert "The system comprises" not in lim.text

    def test_case_insensitive_fallback_match(self):
        claim = Claim(number=1, text=CLAIM_TEXT)
        llm = FakeLLM(responses=[["A widget comprising:", "A MOTOR coupled to the frame."]])
        limitations = split_atomic_limitations(claim, llm=llm)
        assert_verbatim_in_order(limitations, CLAIM_TEXT)
        # re-sliced with the claim's original casing
        assert limitations[1].text == "a motor coupled to the frame."


class TestBuildClaimChart:
    def test_end_to_end_and_coverage_math(self):
        patent = make_patent([Claim(number=1, text="A device comprising: lim A; and lim B.")])
        llm = FakeLLM(
            responses=[
                ["lim A", "lim B"],  # split (verbatim segments of the claim)
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
        # 1 split call + 4 assess calls
        assert len(llm.prompts) == 5

    def test_locator_attaches_citations(self):
        patent = make_patent([Claim(number=1, text="A device comprising: lim A.")])
        llm = FakeLLM(
            responses=[
                ["A device comprising:", "lim A."],
                {"status": "disclosed", "reasoning": "r", "quotes": ["quoted passage"]},
                {"status": "disclosed", "reasoning": "r", "quotes": []},
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
