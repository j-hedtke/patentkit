---
name: claim-chart
description: Build an element-by-element invalidity claim chart mapping a patent claim against prior-art references using the patentkit MCP tools. Use when the user asks for a claim chart, an element-by-element mapping, or coverage of limitations by references.
---

# Claim charting

## Flow

1. **Collect inputs**: the target patent number, the claim number(s) to chart, and the prior-art reference numbers (often the top results of a prior `invalidity-search` run). Verify each is fetchable with `get_patent`; `index_patents` anything missing.
2. **Set expectations**: call `estimate_search_time` with `charting_claims` set to the number of claims — charting is LLM-heavy (~45 s/claim) — and tell the user the estimate before starting.
3. **Chart**: call `build_claim_chart` once per claim with the reference numbers.
4. **Present**: render the chart as a markdown table — one row per atomic limitation, one column per reference, each cell quoting the disclosing passage (with pinpoint context) or marked **NOT DISCLOSED**. Lead with the coverage summary: which limitations are fully covered, which are gaps.
5. **Gaps drive strategy**: for uncovered limitations, offer to run a targeted `invalidity-search` iteration focused on just those limitations (use their language as keywords).

## Notes

- Quote claim and reference language verbatim; never paraphrase inside chart cells.
- If the tool returns an error that the analysis module is unavailable, tell the user to install the full patentkit package (`pip install 'patentkit[all]'`).
