"""Prompt templates for the analysis skills.

Clean-room templates written for patentkit; each is a module constant with
``{placeholder}`` fields filled via ``str.format``. JSON-returning prompts
pair with :meth:`patentkit.llm.base.LLM.complete_json`, which tolerates code
fences around the JSON.

Constants:

- :data:`INTERPRET_CLAIM` — interpret a claim in light of the specification.
- :data:`SPLIT_ATOMIC_LIMITATIONS` — split a claim into atomic limitations.
- :data:`MAP_LIMITATION_SPANS` — map limitations to character spans in the claim.
- :data:`ASSESS_DISCLOSURE` — does a reference disclose one limitation?
- :data:`SELECT_PASSAGES` — verbatim supporting passages for a limitation.
- :data:`KEYWORD_GENERATION` — search keywords from claims + specification.
- :data:`FTO_ANALYSIS` — freedom-to-operate risk of a product vs. one claim.
- :data:`INFRINGEMENT_ANALYSIS` — product evidence vs. one limitation.
- :data:`DRAFT_CLAIMS` — draft claims from an invention disclosure.
- :data:`DRAFT_SPEC_SECTION` — draft one specification section.
"""

from __future__ import annotations

#: Interpret a claim in light of its specification (MEDIUM effort).
#: Placeholders: {specification}, {claim}.
INTERPRET_CLAIM = """\
You are a patent analyst. Below are a patent claim and the specification (or an
excerpt) from the same patent.

<specification>
{specification}
</specification>

<claim>
{claim}
</claim>

Interpret the claim in light of the specification. Where the specification clearly
defines or explains a claim term, annotate that term inline by appending a short
parenthetical giving the specification-supported, plain-language meaning. If a term
is not specially defined, leave it unannotated — most claim terms carry their plain
and ordinary meaning. Return the full claim text with any annotations, preserving
the original claim structure. Do not add commentary outside the claim text."""


#: Split a claim into atomic limitations as a JSON list of strings (HIGH effort).
#: Placeholders: {claim}.
SPLIT_ATOMIC_LIMITATIONS = """\
Split the following patent claim into atomic limitations — the smallest separately
assessable requirements of the claim.

<claim>
{claim}
</claim>

Rules:
1. Each limitation must be a VERBATIM, contiguous segment of the claim text.
   Copy it exactly as written — do NOT paraphrase, reword, abbreviate, expand,
   or normalize punctuation or capitalization.
2. List the limitations in the order they appear in the claim (document order):
   the preamble segment first (everything up to and including "comprising:" or
   the equivalent transitional phrase), then each claim element verbatim, in
   order.
3. Do not reorder segments, and do not merge text across semicolons unless a
   single element genuinely spans them.
4. Together the segments must cover essentially ALL of the claim language,
   including the preamble; do not drop any clause.

Return ONLY a JSON list of strings, e.g. ["limitation 1", "limitation 2"]."""


#: Map each limitation to a character span in the claim text (LOW effort).
#: Placeholders: {claim_text}, {limitations} (a JSON list of strings).
MAP_LIMITATION_SPANS = """\
Here is a patent claim and a list of atomic limitations derived from it.

<claim>
{claim_text}
</claim>

<limitations>
{limitations}
</limitations>

For each limitation, identify the character span of the claim text that best
expresses it. Spans are 0-based character offsets into the claim text exactly as
given above. Spans must not overlap, must start and end on word boundaries, and
together should cover as much of the claim text as possible.

Return ONLY a JSON list of objects:
[{{"limitation": "<the limitation verbatim>", "start": <int>, "end": <int>}}, ...]"""


#: Assess whether a reference discloses one limitation (HIGH effort).
#: Placeholders: {reference}, {limitation}.
ASSESS_DISCLOSURE = """\
Here is a prior-art reference text:

<reference>
{reference}
</reference>

And one atomic claim limitation:

<limitation>
{limitation}
</limitation>

Determine whether the limitation is disclosed by the reference, considering both
explicit disclosure and what a person of ordinary skill would clearly infer.

- "disclosed": every requirement of the limitation is taught or clearly inferable.
- "partial": some but not all requirements are taught.
- "not_disclosed": the reference does not teach the limitation.

Return ONLY a JSON object:
{{"status": "disclosed" | "partial" | "not_disclosed",
  "reasoning": "<why, citing what the reference does or does not teach>",
  "quotes": ["<verbatim passages from the reference supporting your finding>"]}}

Quotes must be copied verbatim from the reference text, unaltered. Use an empty
list when no passage supports the finding."""


#: Select verbatim supporting passages for a disclosure finding (MEDIUM effort).
#: Placeholders: {reference}, {limitation}, {reasoning}.
SELECT_PASSAGES = """\
Here is a prior-art reference text:

<reference>
{reference}
</reference>

A prior analysis concluded the following about this atomic claim limitation:

<limitation>{limitation}</limitation>
<reasoning>{reasoning}</reasoning>

Identify every passage of the reference that supports this conclusion. Copy each
passage VERBATIM from the reference — do not paraphrase, truncate mid-sentence, or
alter punctuation. Include as many passages as are genuinely relevant.

Return ONLY a JSON list of objects:
[{{"passage": "<verbatim passage>", "reasoning": "<how it supports the finding>"}}, ...]

Return an empty JSON list if no passage supports the conclusion."""


#: Generate search keywords from claims + specification (LOW effort, voted).
#: Placeholders: {claims}, {description}.
KEYWORD_GENERATION = """\
Here are the claims of a patent (or a technology description) and supporting
description text:

<claims>
{claims}
</claims>

<description>
{description}
</description>

Propose 15-20 search keywords or short phrases for finding related patents.

Guidelines:
1. Focus on the technical features and components that make this technology distinctive.
2. Cover the core functionality, not just the structure.
3. Include common synonyms and alternative terms of art.
4. Prefer terms that recur or seem central to the invention.
5. Avoid overly narrow or overly generic terms — these drive a coarse first-pass search.

Return ONLY a JSON list of strings."""


#: Freedom-to-operate analysis of a product against one claim (HIGH effort).
#: Placeholders: {product_description}, {patent_number}, {claim}.
FTO_ANALYSIS = """\
Here is a description of a product or planned technology:

<product>
{product_description}
</product>

And claim text from patent {patent_number}:

<claim>
{claim}
</claim>

Analyze freedom-to-operate risk in steps:

1. Literal reading: does the product practice every element of the claim exactly as
   recited? Explain element by element.
2. If not literal, doctrine of equivalents:
   a) Known interchangeability — is any differing component a known interchangeable
      substitute that would yield predictable results?
   b) Function-way-result — does the product's component perform substantially the
      same function, in substantially the same way, to achieve substantially the
      same result as the claimed element?

Return ONLY a JSON object:
{{"risk": "literal" | "doe" | "none",
  "confidence": 1 | 2 | 3,
  "assumptions": "<the key assumptions your analysis rests on — always required>",
  "argument": "<your element-by-element analysis; empty string if risk is \\"none\\">"}}

Confidence: 1 = low, 2 = medium, 3 = high. "doe" covers infringement found only
under either doctrine-of-equivalents theory."""


#: Assess product evidence against one limitation (HIGH effort).
#: Placeholders: {product_name}, {limitation}, {evidence}.
INFRINGEMENT_ANALYSIS = """\
You are analyzing whether the product "{product_name}" meets one limitation of a
patent claim, based on collected public evidence.

<limitation>
{limitation}
</limitation>

Evidence (each block is tagged with its source):

<evidence_corpus>
{evidence}
</evidence_corpus>

Determine whether the evidence shows the product meets the limitation:

- "met": the evidence directly shows the product practices the limitation.
- "likely": the evidence strongly suggests it, but a detail is unconfirmed.
- "unclear": the evidence is insufficient to tell either way.
- "not_met": the evidence shows the product does NOT practice the limitation.

Return ONLY a JSON object:
{{"status": "met" | "likely" | "unclear" | "not_met",
  "reasoning": "<your analysis tying the evidence to the limitation language>",
  "evidence": [{{"source": "<source tag/url>", "quote": "<verbatim quote>",
                 "note": "<what this quote establishes>"}}, ...]}}

Quotes must be verbatim from the evidence corpus. Use an empty evidence list when
nothing in the corpus bears on the limitation."""


#: Draft patent claims from an invention disclosure (HIGH effort).
#: Placeholders: {disclosure}, {n_independent}, {n_dependent}.
DRAFT_CLAIMS = """\
You are a patent attorney drafting claims. Here is the invention disclosure:

<disclosure>
{disclosure}
</disclosure>

Draft {n_independent} independent claim(s) and {n_dependent} dependent claim(s).

Drafting rules:
1. Use standard US claim format: each claim is one sentence, numbered "1.", "2.", ...
2. Independent claims: a preamble ending in "comprising:", then elements separated
   by semicolons, with "wherein" clauses for functional relationships.
3. Dependent claims reference their parent ("The ... of claim N, wherein ...") and
   add exactly one meaningful narrowing feature each.
4. Maintain strict antecedent basis: introduce features with "a"/"an", refer back
   with "the".
5. Focus on technical features; avoid purely business or mental steps.

Return ONLY the numbered claims, nothing else."""


#: Draft one specification section from a disclosure (HIGH effort).
#: Placeholders: {section}, {disclosure}, {claims}.
DRAFT_SPEC_SECTION = """\
You are a patent attorney drafting the "{section}" section of a patent application.

Invention disclosure:

<disclosure>
{disclosure}
</disclosure>

Claims drafted so far (may be empty):

<claims>
{claims}
</claims>

Draft the {section} section in conventional patent-specification style. Ensure
every claim term is supported by the description, use reference-numeral-free prose,
and keep the level of detail appropriate to the section. Return ONLY the section
text, without the section heading."""
