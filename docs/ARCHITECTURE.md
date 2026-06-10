# patentkit architecture

## Layering (strict, bottom-up)

```
integrations (MCP / OpenAI tools / Claude plugin / Word add-in)
    └── agents (guided search, invalidity/fto/infringement search, charting)
        ├── analysis (LLM skills) ── formatting (charts, reports)
        ├── search (keyword + vector stores, hybrid fusion)
        ├── connectors (inference / infra / training)
        └── notify, viz, evals
            └── llm (effort-routed providers)  ── parsing
                └── models (canonical Patent)  ── config (keys)
```

Rules:

- Lower layers never import higher layers.
- Optional heavy dependencies (`elasticsearch`, `pymupdf`, `python-docx`,
  `sklearn`, `mcp`, provider SDKs, `bs4`) are imported lazily inside functions
  with errors that name the pip extra to install.
- Higher layers import sibling capabilities lazily and degrade: agents work
  without `analysis` (skip charting), without a vector store (keyword only),
  and without an LLM (heuristic keywords, no rerank).

## Canonical data model

`patentkit.models.Patent` is the single interchange type. Connectors attach a
`SourceRecord(source, fidelity)`; `Patent.merge()` reconciles duplicate records:
higher fidelity wins per scalar field, present beats absent, list fields union
with entity dedup, citation origin flags (examiner / applicant / third-party /
family) are OR-ed. Source fidelity convention: google_patents=3,
uspto_odp=2, uspto_bulk=2, epo_ops=1, serpapi=1.

## Search

`SearchQuery` is the one query type for every backend and is deliberately the
full user-facing parameter surface: keywords, required/excluded tokens,
minimum-match, fields, CPC art classes, inventors, assignees, date cutoffs,
countries, allow/deny number lists, limit. Backends apply what they support;
`apply_metadata_filters` gives in-memory backends the reference semantics.

Staged retrieval (mirrors the production pipeline patentkit derives from):

1. **Stage 1 — recall**: BM25/ES keyword search (k≈1000), prior-art date
   cutoff, default exclusions.
2. **Stage 2 — precision**: vector similarity + RRF fusion (k=60).
3. **Stage 3 — LLM relevance**: batched scoring of top candidates against the
   asserted claims; final score `0.75·llm + 0.25·retrieval` (z-normalized).

Default invalidity exclusions: the target patent, its family, examiner-cited
art (from `Patent.citations` and, when available, the USPTO file wrapper), and
user-supplied custom exclusions. All overridable.

## LLM routing

Skills declare `ReasoningEffort`, not models:

| effort | used for | anthropic default | openai default |
|---|---|---|---|
| low | extraction, keyword voting, topic naming | claude-haiku-4-5 | gpt-5-mini (effort=low) |
| medium | claim interpretation, passage selection | claude-sonnet-4-6 | gpt-5.1 (effort=medium) |
| high | disclosure assessment, planning, charting, drafting | claude-fable-5 | gpt-5.1 (effort=high) |

Override per call (`get_llm(model=...)`) or globally (`DEFAULT_MODELS`).
`claude-opus-4-8` is registered as the high-effort Anthropic alternate.

## PDF line-number regression

Issued US patents print line markers (5, 10, 15…) in the column gutter.
The old approach interpolated between detected markers and snapped to the
nearest point; OCR noise (missing/misread markers) propagated directly into
citations. patentkit instead fits **least-squares `line = a·y + b` per
(page, column) with iterative outlier rejection** (drop |residual| >
max(2σ, 0.5 line), refit). A misread marker becomes an outlier to discard
rather than a point to interpolate through, and predictions remain stable for
y-positions outside the marker range. `locate_passage()` fuzzy-matches text,
then maps endpoints through the model to emit "col. 3, ll. 45–52" citations.

## Guided sessions

`GuidedSearchSession` is a serializable state machine
(`planning → awaiting_plan_feedback → searching → awaiting_result_feedback →
done`) so multi-turn agent harnesses (MCP tools, OpenAI tool calls, Claude
skills) can drive plan → feedback → execute → iterate across turns. Feedback is
structured at three granularities: whole result, passage, query.

## Exposure

`integrations.toolset.PatentToolset` + `TOOL_SPECS` is the single tool
surface; the MCP server and the OpenAI function-tool layer are both thin
wrappers over it, so the two ecosystems stay in lockstep.
