---
name: invalidity-search
description: Run a guided prior-art (invalidity) search against a patent using the patentkit MCP tools. Use when the user wants to invalidate a patent, find prior art, anticipate/obviousness references, or asks "what came before patent X".
---

# Guided invalidity search

Drive the patentkit MCP tools through the guided loop. Never skip the plan-review step unless the user explicitly asks for a fully autonomous run.

## Flow

1. **Identify the target.** Get the patent number (e.g. `US10123456B2`) and which claims matter (default: independent claims). Call `get_patent` to confirm the record exists and show the user the title + claim 1 so they can confirm it is the right patent. If it is missing, `index_patents` with the number or ask for a corpus JSONL.
2. **Start the session.** Call `guided_search_start` with `search_type="invalidity"`, the patent number, and the claim list. **Present the returned plan and the estimated time up front** (the response includes `estimated_seconds` / `estimated_human`): show each planned query's purpose and keywords, the prior-art date cutoff, and the exclusion list.
3. **Examiner-art exclusion (important).** By default the pipeline excludes examiner-cited art, family members, and the patent itself — tell the user this. If they want examiner-cited references included (e.g. to re-argue cited art), collect that as feedback and note the exclusions can be overridden by removing numbers from the plan's exclusions or by running the Python API with `exclude_examiner_art=False`.
4. **Collect plan feedback.** Ask the user to judge each query (`good` / `too_broad` / `too_narrow` / `off_topic`), add free-text guidance, then call `guided_search_feedback`. Show the revised plan; repeat until they approve.
5. **Execute.** Call `guided_search_execute`. Remind the user of the time estimate before running.
6. **Present results.** Show the ranked references as a numbered list: patent number, title, score, the *why* sentence, and the **highlighted passages** (quote 1-2 best passages per reference). Also summarize what was excluded and why (`excluded` map in the response).
7. **Iterate.** Ask the user to mark results relevant/irrelevant and note missing angles; send that via `guided_search_feedback` (irrelevant results become exclusions) and re-execute. Stop when the user is satisfied.
8. **Optional charting.** Offer `build_claim_chart` against the top 2-4 references for the key claims.

## Tips

- Keep claim language verbatim when discussing limitations.
- If results look thin, suggest broadening: lower minimum-match, synonyms, adjacent CPC classes.
- The date cutoff is the patent's earliest priority/filing date; never present post-cutoff documents as prior art.
