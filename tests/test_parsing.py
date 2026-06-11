"""Tests for pure-python claim parsing, deterministic limitation splitting,
claims-section extraction, and stdlib HTML-to-text conversion."""

from patentkit.models import Claim, Limitation
from patentkit.parsing import (
    claim_element_outline,
    extract_claims_section,
    html_to_text,
    parse_claims,
    split_limitations,
)

FIVE_CLAIMS = """\
1. A beverage container comprising:
a body defining an interior chamber;
a lid removably coupled to the body, the lid including a vent; and
wherein the vent is configured to release steam from the interior chamber.
2. The beverage container of claim 1, wherein the body comprises stainless steel.
3 . The beverage container of claim 2, further comprising a handle attached to the body.
4. A method of brewing a beverage, the method comprising:
heating water in a reservoir;
and dispensing the water over grounds, wherein the dispensing occurs at a controlled rate.
5. The method of any of claims 4, wherein the water is heated to a boil.
"""


class TestParseClaims:
    def test_parses_all_five_claims(self):
        claims = parse_claims(FIVE_CLAIMS)
        assert [c.number for c in claims] == [1, 2, 3, 4, 5]

    def test_dependencies(self):
        claims = parse_claims(FIVE_CLAIMS)
        assert [c.depends_on for c in claims] == [None, 1, 2, None, 4]
        assert claims[0].is_independent
        assert not claims[1].is_independent
        assert claims[3].is_independent

    def test_number_space_period_style(self):
        # claim "3 ." uses the spaced style
        claims = parse_claims(FIVE_CLAIMS)
        assert claims[2].number == 3
        assert "handle" in claims[2].text

    def test_element_tree_with_preamble(self):
        claims = parse_claims(FIVE_CLAIMS)
        claim1 = claims[0]
        assert len(claim1.elements) == 1  # preamble is the root
        root = claim1.elements[0]
        assert root.text.endswith("comprising:")
        assert len(root.children) == 3
        assert root.children[0].text.startswith("a body")

    def test_wherein_clause_becomes_child(self):
        claims = parse_claims(FIVE_CLAIMS)
        claim4 = claims[3]
        root = claim4.elements[0]
        dispensing = root.children[1]
        assert dispensing.text.startswith("dispensing the water")
        assert len(dispensing.children) == 1
        assert dispensing.children[0].text.startswith("wherein the dispensing")

    def test_outline_rendering(self):
        claims = parse_claims(FIVE_CLAIMS)
        outline = claim_element_outline(claims[0])
        assert outline.startswith("Claim 1:")
        assert "  - A beverage container comprising:" in outline
        assert "    - a body defining an interior chamber" in outline


MULTI_ELEMENT_CLAIM = (
    "A computer-based system for detecting structures in data, comprising: "
    "an input device for receiving data; "
    "an output device for presenting the data; "
    "a memory storing program routines, wherein the routines include an "
    "analyzer server for detecting structures in the data; and "
    "a processing unit coupled to the input device, the output device, and "
    "the memory."
)


class TestSplitLimitations:
    """The deterministic structural splitter — the PRIMARY mechanism for
    producing limitation units, run at parse/index time without an LLM."""

    def _assert_verbatim_spans(self, limitations, claim_text):
        for lim in limitations:
            assert lim.span is not None
            start, end = lim.span
            assert claim_text[start:end] == lim.text
        starts = [lim.span[0] for lim in limitations]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_multi_element_claim_labels_and_segments(self):
        limitations = split_limitations(MULTI_ELEMENT_CLAIM, 1)
        self._assert_verbatim_spans(limitations, MULTI_ELEMENT_CLAIM)
        assert [lim.label for lim in limitations] == [
            "1[pre]", "1[a]", "1[b]", "1[c]", "1[d]",
        ]
        assert limitations[0].text.endswith("comprising:")
        assert limitations[1].text == "an input device for receiving data;"
        # "; and" delimiter stays with the preceding element
        assert limitations[3].text.endswith("structures in the data; and")
        assert limitations[4].text.startswith("a processing unit coupled")
        # together the segments cover essentially the whole claim
        joined = " ".join(lim.text for lim in limitations)
        assert " ".join(joined.split()) == " ".join(MULTI_ELEMENT_CLAIM.split())

    def test_consisting_of_transition_ends_preamble(self):
        text = "An alloy consisting of: iron; nickel; and chromium."
        limitations = split_limitations(text, 2)
        self._assert_verbatim_spans(limitations, text)
        assert limitations[0].label == "2[pre]"
        assert limitations[0].text == "An alloy consisting of:"
        assert [lim.text for lim in limitations[1:]] == [
            "iron;", "nickel; and", "chromium.",
        ]
        assert [lim.label for lim in limitations[1:]] == ["2[a]", "2[b]", "2[c]"]

    def test_nested_wherein_clause_stays_with_its_element(self):
        # wherein clauses do not split elements: only ";" boundaries do
        limitations = split_limitations(MULTI_ELEMENT_CLAIM, 1)
        memory = next(lim for lim in limitations if lim.text.startswith("a memory"))
        assert "wherein the routines include" in memory.text

    def test_claim_with_no_comprising_is_single_limitation(self):
        text = "A widget made of solid brass."
        limitations = split_limitations(text, 3)
        assert len(limitations) == 1
        assert limitations[0].label == "3[a]"
        assert limitations[0].text == text
        assert limitations[0].span == (0, len(text))

    def test_no_preamble_but_semicolons_still_split(self):
        text = "A kit including a bolt; a nut; and a washer."
        limitations = split_limitations(text, 4)
        self._assert_verbatim_spans(limitations, text)
        assert [lim.label for lim in limitations] == ["4[a]", "4[b]", "4[c]"]

    def test_whitespace_trimmed_but_spans_index_original_text(self):
        text = "A device comprising:   a frame;   and a motor."
        limitations = split_limitations(text, 1)
        self._assert_verbatim_spans(limitations, text)
        assert limitations[1].text == "a frame;   and"
        assert limitations[2].text == "a motor."

    def test_empty_text_yields_no_limitations(self):
        assert split_limitations("", 1) == []

    def test_parse_claims_populates_limitations(self):
        claims = parse_claims(FIVE_CLAIMS)
        claim1 = claims[0]
        assert claim1.limitations  # populated at parse time
        assert claim1.limitations[0].label == "1[pre]"
        assert claim1.limitations[0].text.endswith("comprising:")
        assert claims[3].limitations[0].label == "4[pre]"


class TestLazyLimitationAccessor:
    """Claim.get_limitations() computes-and-caches for records serialized
    before limitations existed (e.g. the eval corpus JSONL)."""

    def test_deserialized_claim_without_limitations_computes_lazily(self):
        claim = Claim.model_validate_json(
            '{"number": 1, "text": "A device comprising: a frame; and a motor."}'
        )
        assert claim.limitations == []  # nothing stored
        limitations = claim.get_limitations()
        assert [lim.label for lim in limitations] == ["1[pre]", "1[a]", "1[b]"]
        # cached on the model: same objects, no recompute
        assert claim.limitations is limitations
        assert claim.get_limitations() is limitations

    def test_old_serialized_records_with_extra_fields_still_load(self):
        # pre-migration records carried "atomic_limitations"; pydantic
        # ignores the unknown key and the accessor recomputes
        claim = Claim.model_validate_json(
            '{"number": 2, "text": "A widget comprising: a gear.",'
            ' "atomic_limitations": [{"text": "a gear.", "span": [21, 28]}]}'
        )
        assert claim.limitations == []
        assert [lim.label for lim in claim.get_limitations()] == ["2[pre]", "2[a]"]

    def test_preset_limitations_are_never_overwritten(self):
        preset = [Limitation(label="1[a]", text="a frame; and a motor.")]
        claim = Claim(
            number=1, text="A device comprising: a frame; and a motor.",
            limitations=preset,
        )
        assert claim.get_limitations() == preset


class TestExtractClaimsSection:
    def test_what_is_claimed_anchor(self):
        spec = (
            "BACKGROUND\nPrior containers leak.\n"
            "DETAILED DESCRIPTION\nThe container 10 includes a body 12.\n"
            "What is claimed is:\n" + FIVE_CLAIMS
        )
        section = extract_claims_section(spec)
        assert section.startswith("1. A beverage container")
        assert "BACKGROUND" not in section

    def test_we_claim_anchor(self):
        spec = "Some description.\nWe claim:\n1. A widget comprising a frame."
        assert extract_claims_section(spec).startswith("1. A widget")

    def test_claims_heading_anchor(self):
        spec = "Description here.\nCLAIMS\n1. A widget comprising a frame."
        assert extract_claims_section(spec).startswith("1. A widget")

    def test_no_anchor_returns_input(self):
        text = "1. A widget comprising a frame."
        assert extract_claims_section(text) == text


class TestHtmlToText:
    def test_strips_scripts_and_keeps_text(self):
        html = (
            "<html><head><style>p { color: red; }</style>"
            "<script>var leak = 1;</script></head>"
            "<body><h1>Patent Title</h1><p>Hello <b>world</b>.</p>"
            "<div>Second block</div></body></html>"
        )
        text = html_to_text(html)
        assert "Patent Title" in text
        assert "Hello world." in text
        assert "Second block" in text
        assert "var leak" not in text
        assert "color: red" not in text

    def test_entities_decoded(self):
        assert "A & B" in html_to_text("<p>A &amp; B</p>")
