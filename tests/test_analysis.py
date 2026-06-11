"""Tests for the analysis skills using FakeLLM (no network, no extras)."""

from patentkit.analysis import (
    build_claim_chart,
    check_antecedent_basis,
    generate_keywords,
    refine_limitations,
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


class TestRefineLimitations:
    """refine_limitations is an OPTIONAL LLM pass over the precomputed
    deterministic units; verbatimness and ordering are enforced in code."""

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
        limitations = refine_limitations(claim, llm=llm)
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        # (c) preamble first, then the first element after the preamble
        assert limitations[0].text == DETECT_PREAMBLE
        assert limitations[1].text == DETECT_ELEMENT_1
        # whitespace-sloppy segment was re-sliced verbatim from the claim
        assert limitations[2].text == "an output device for presenting the data;"
        # relabeled in order: preamble keeps [pre], elements get letters
        assert [lim.label for lim in limitations] == ["1[pre]", "1[a]", "1[b]"]
        assert len(llm.prompts) == 1  # single refine call, no span-mapping call
        # the deterministic units were offered to the LLM as the baseline
        assert DETECT_ELEMENT_1 in llm.prompts[0]

    def test_llm_none_returns_precomputed_limitations(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        limitations = refine_limitations(claim, llm=None)
        assert limitations == claim.get_limitations()
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)

    def test_merging_segments_is_allowed_and_relabeled(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        merged = (
            "an input device for receiving data; an output device for "
            "presenting the data;"
        )
        llm = FakeLLM(responses=[[DETECT_PREAMBLE, merged]])
        limitations = refine_limitations(claim, llm=llm)
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        assert [lim.label for lim in limitations] == ["1[pre]", "1[a]"]
        assert limitations[1].text == merged

    def test_paraphrasing_llm_falls_back_to_deterministic_units(self):
        claim = Claim(number=1, text=DETECT_CLAIM_TEXT)
        llm = FakeLLM(
            responses=[
                [
                    "The system comprises an input device for receiving data.",
                    "There is an output device.",
                ]
            ]
        )
        limitations = refine_limitations(claim, llm=llm)
        # paraphrases never survive: result is verbatim, in order
        assert_verbatim_in_order(limitations, DETECT_CLAIM_TEXT)
        assert limitations[0].text == DETECT_PREAMBLE
        assert limitations[1].text == DETECT_ELEMENT_1
        for lim in limitations:
            assert "The system comprises" not in lim.text

    def test_case_insensitive_fallback_match(self):
        claim = Claim(number=1, text=CLAIM_TEXT)
        llm = FakeLLM(responses=[["A widget comprising:", "A MOTOR coupled to the frame."]])
        limitations = refine_limitations(claim, llm=llm)
        assert_verbatim_in_order(limitations, CLAIM_TEXT)
        # re-sliced with the claim's original casing
        assert limitations[1].text == "a motor coupled to the frame."


class TestBuildClaimChart:
    def test_end_to_end_and_coverage_math(self):
        # Rows are the PRECOMPUTED structural units: preamble, "lim A; and",
        # "lim B." — the LLM is only used for disclosure assessment.
        patent = make_patent([Claim(number=1, text="A device comprising: lim A; and lim B.")])
        llm = FakeLLM(
            responses=[
                # reference US111: preamble + lim A disclosed, lim B not
                {"status": "disclosed", "reasoning": "device taught", "quotes": []},
                {"status": "disclosed", "reasoning": "teaches A", "quotes": ["the art shows A"]},
                {"status": "not disclosed", "reasoning": "silent on B", "quotes": []},
                # reference US222: preamble + lim B disclosed, lim A not
                {"status": "disclosed", "reasoning": "device taught", "quotes": []},
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
        assert [lim.text for lim in chart.limitations] == [
            "A device comprising:", "lim A; and", "lim B.",
        ]
        assert [lim.label for lim in chart.limitations] == ["1[pre]", "1[a]", "1[b]"]
        assert len(chart.references) == 2
        assert chart.references[0].findings[1].status == "disclosed"
        assert chart.references[0].findings[1].quotes == ["the art shows A"]
        assert chart.references[0].findings[2].status == "not_disclosed"

        # coverage math: each reference discloses 2 of 3 limitations
        assert chart.coverage_summary() == {"US111": 2 / 3, "US222": 2 / 3}
        # but together they cover all three
        assert chart.combined_coverage() == 1.0
        # NO split call — 6 assess calls only
        assert len(llm.prompts) == 6

    def test_locator_attaches_citations(self):
        patent = make_patent([Claim(number=1, text="A device comprising: lim A.")])
        llm = FakeLLM(
            responses=[
                {"status": "disclosed", "reasoning": "r", "quotes": ["quoted passage"]},
                {"status": "disclosed", "reasoning": "r", "quotes": []},
            ]
        )
        chart = build_claim_chart(
            patent, 1, references=[("US111", "ref text")],
            llm=llm, locator=lambda passage: "col. 3, ll. 45-52",
        )
        assert chart.references[0].findings[0].citation == "col. 3, ll. 45-52"

    def test_precomputed_limitations_are_respected(self):
        # A pre-populated claim.limitations (e.g. from refine_limitations)
        # is used as-is — never re-split.
        from patentkit.models import Limitation

        claim = Claim(
            number=1, text="A device comprising: lim A; and lim B.",
            limitations=[Limitation(label="1[a]", text="lim A; and lim B.")],
        )
        patent = make_patent([claim])
        llm = FakeLLM(responses=[
            {"status": "disclosed", "reasoning": "r", "quotes": []},
        ])
        chart = build_claim_chart(patent, 1, references=[("US111", "ref text")], llm=llm)
        assert [lim.label for lim in chart.limitations] == ["1[a]"]
        assert len(llm.prompts) == 1  # one assess call for the single unit

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
