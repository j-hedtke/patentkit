---
name: invalidity-search
description: Run a guided prior-art (invalidity) search against a patent using the patentkit MCP tools. Use when the user wants to invalidate a patent, find prior art, anticipate/obviousness references, or asks "what came before patent X".
---

# Guided invalidity search

Drive the patentkit MCP tools through the guided loop. Execution is agentic: one LLM agent iteratively generates search queries, runs them as tools, reads the results, refines its angles, and finishes with ranked candidates — under a time budget, with a full saved reasoning trace. Never skip the plan-review step unless the user explicitly asks for a fully autonomous run.

## Flow

1. **Identify the target.** Get the patent number (e.g. `US10123456B2`) and which claims matter (default: independent claims). Call `get_patent` to confirm the record exists and show the user the title + claim 1 so they can confirm it is the right patent. If it is missing, `index_patents` with the number or ask for a corpus JSONL.
2. **Start the session.** Call `guided_search_start` with `search_type="invalidity"`, the patent number, and the claim list. **Present the returned plan preview and the estimated time up front** (the response includes `estimated_seconds` / `estimated_human`). The plan is a preview of *starting angles* — the agent generates and refines its own queries during execution — plus the prior-art date cutoff and the exclusion list.
3. **Examiner-art exclusion (important).** By default the tool layer hard-excludes examiner-cited art, family members, and the patent itself (excluded numbers can never appear in the agent's search results) — tell the user this. If they want examiner-cited references included (e.g. to re-argue cited art), run the Python API with `exclude_examiner_art=False`.
4. **Collect plan feedback.** Ask the user to judge each starting angle (`good` / `too_broad` / `too_narrow` / `off_topic`), add free-text guidance, then call `guided_search_feedback`. Pre-run feedback adjusts the preview and is seeded into the agent's initial guidance; repeat until they approve.
5. **Execute.** Call `guided_search_execute`. Remind the user of the time estimate before running. The response includes `stop_reason` and a `trace_summary` (step count, the queries the agent issued, its intermediate shortlist).
6. **Show the trace.** Call `get_search_trace` and show the user the queries the agent issued and how its shortlist evolved (the markdown trace is the agent's full reasoning). This is the basis for feedback on *queries*, not just results.
7. **Present results.** Show the ranked references as a numbered list: patent number, title, confidence, the *why* sentence, and the **key passages** (quote 1-2 per reference). Also summarize what was excluded and why (`excluded` map in the response).
8. **Iterate.** Collect feedback on both the queries and the results (mark results relevant/irrelevant, note missing angles); send it via `guided_search_feedback` and re-execute. Feedback is injected as a user message into the SAME resumed agent conversation (irrelevant results also become hard exclusions). Stop when the user is satisfied.
9. **Optional charting.** Offer `build_claim_chart` against the top 2-4 references for the key claims.

## Tips

- Keep claim language verbatim when discussing limitations.
- If results look thin, relay that to the agent as feedback: suggest synonyms, adjacent CPC classes, component-level phrasing — it will issue new queries on resume.
- The date cutoff is the patent's earliest priority/filing date and is enforced at the tool layer; never present post-cutoff documents as prior art.
- Without an LLM configured the server runs a clearly-labeled degraded single keyword pass (no trace); say so if `stop_reason` is `degraded`.
