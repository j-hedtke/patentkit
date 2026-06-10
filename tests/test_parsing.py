"""Tests for pure-python claim parsing, claims-section extraction, and
stdlib HTML-to-text conversion."""

from patentkit.parsing import (
    claim_element_outline,
    extract_claims_section,
    html_to_text,
    parse_claims,
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
