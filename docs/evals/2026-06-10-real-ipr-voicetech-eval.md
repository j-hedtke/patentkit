# Invalidity search eval — real IPR ground truth (IPR2020-01018, Voice Tech)

**Date:** 2026-06-10 · **Index:** `patentkit-eval-corpus` (single-node ES 8.14.3)
· **Corpus:** 300 real patents live-scraped from Google Patents
(`scripts/build_eval_corpus.py`), distractors drawn from the citation
neighborhood of the target and references (hard negatives by construction).

## The queryset is a real adjudicated IPR

Unlike the earlier toy set (see `2026-06-10-ipr-es-eval.md`), the ground truth
here is what the PTAB actually held:

- **Proceeding:** IPR2020-01018, *Unified Patents, LLC v. Voice Tech Corp.*
- **Patent:** US10491679B2 — "Using voice commands from a mobile device to
  remotely access and control a computer" (priority 2007-06-04)
- **Outcome:** Final Written Decision (2021-12-13) held claims 1–8 obvious
  over **Wong (US20060235700A1)** in view of **Beauregard (US6438545B1)**;
  affirmed, *Voice Tech Corp. v. Unified Patents, LLC*, No. 22-2163
  (Fed. Cir. Aug. 1, 2024) (precedential).
- **Queryset:** target US10491679B2, claim 1; ground truth = the two
  references of the winning ground.

## Eval-design lesson: examiner-cited art must not be excluded here

Wong sits **on the face of the '679 patent** (examiner-cited). patentkit's
product default excludes examiner-cited art at the tool layer — correct for
prosecution-style searching, but IPR petitioners may (and here did) build the
winning ground on face-of-patent art. With the default exclusion the agent
can never return Wong and recall is silently floored at 0. The eval therefore
runs `exclude_examiner_art=False` and records that in the report. Any
IPR-derived eval must check its ground truth against the exclusion lists.

## Results — keys-free baseline (degraded keyword-only mode — NOT agentic)

| Metric | recall@5 | recall@10 | recall@25 | recall@50 | MRR |
|---|---|---|---|---|---|
| baseline | 0.50 | 0.50 | 1.00 | 1.00 | 0.333 |

Both references retrieved, ranked 3rd and ~11th–25th. Report:
`eval_report.keysfree-baseline.json`.

## Results — live agentic search (Anthropic, medium-effort routing)

| Metric | recall@5 | recall@10 | recall@25 | recall@50 | MRR |
|---|---|---|---|---|---|
| agentic | 0.50 | 0.50 | 0.50 | 0.50 | **1.000** |

The agent put **Wong — the primary reference of the winning ground — at
rank 1** with an on-point rationale, but did not surface Beauregard in its
20 candidates before the 180s budget expired (`stop_reason:
budget_exceeded`). Reasoning trace: `trace_US10491679B2.json`.

Read together: the agentic mode is a high-precision shortlister (the thing a
practitioner reads first is the right reference); the keyword pass is a
high-recall dragnet. The product flow runs both — and guided feedback (see
`examples/demo_guided_search.py`) is the lever for recovering the secondary
reference the agent missed.

## Harness fix shaken out by this eval

The first agentic run scored 0.0 because the model answered the
budget-exhausted wrap-up with an **empty** `finish` call; the loop's single
grace round ended and the run produced zero candidates. `run_tool_loop` now
grants exactly one retry when the finish call itself errors
(`src/patentkit/llm/tools.py`, regression-tested).

## Repro

```sh
scripts/eval_e2e.sh        # or: /eval-e2e from Claude Code
```
