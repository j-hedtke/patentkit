---
name: patent-drafting
description: Draft patent claims and specification sections informed by prior-art searches via the patentkit MCP tools. Use when the user wants to draft or revise claims, design around prior art, or prepare an application disclosure.
---

# Prior-art-informed patent drafting

## Flow

1. **Understand the invention.** Collect a disclosure: what problem it solves, how it works, what is believed novel. Restate the inventive concept in one sentence and confirm.
2. **Search before drafting.** Run a prior-art check with the patentkit tools: `search_patents` with the invention's key terms (or a full guided `invalidity-search`-style session with `guided_search_start`, treating the draft claim as the target). Present the closest references with their highlighted passages.
3. **Draft claims around the art.**
   - Draft independent claim 1 to the broadest scope that clears the closest references; identify the distinguishing limitation explicitly.
   - Add dependent claims layering fallback positions (each should add one meaningful limitation).
   - Use consistent antecedent basis ("a widget ... the widget"), avoid means-plus-function unless intended, and keep one statutory class per claim.
4. **Validate the draft.** Re-run `search_patents` using the draft claim's distinctive terms to check nothing closer surfaces; iterate the claim language if it does.
5. **Specification support.** For each claim term, draft specification language providing support and at least one alternative embodiment (breadth for later amendments).
6. **Deliver**: the claim set, a short novelty argument over the closest 2-3 references found (cite their numbers and the passages), and a list of terms needing definitional support.

## Caveat

Drafts are starting points for a registered practitioner — say so. Never present a filing-ready opinion on patentability.
