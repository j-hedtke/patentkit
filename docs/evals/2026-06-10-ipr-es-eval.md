# Invalidity search eval — IPR-example corpus on managed Elasticsearch (GCP)

**Date:** 2026-06-10 · **Index:** `patentkit-eval-corpus` (single-node ES 8.14.3
on GKE, project `careful-ai`, cluster `patentkit-eval`) · **Corpus:** 300 real
patents live-scraped from Google Patents (`scripts/build_eval_corpus.py`).

## Setup

Two IPR-example queries seeded the corpus; distractors were selected by
citation-graph BFS around the targets and ground-truth references, so every
distractor is topically adjacent:

| Query patent | Subject | Ground-truth prior art |
|---|---|---|
| US5946647A (Apple '647, data detectors) | action on detected text structures | US5644735A, US5859636A |
| US8046721B2 (slide-to-unlock) | gesture unlock on touch display | US6209104B1, US5821933A |

Pipeline: `scripts/build_eval_corpus.py` (scrape) → `scripts/index_eval_corpus.py`
(ES index) → `scripts/run_invalidity_eval.py` (InvaliditySearchAgent → recall@k/MRR).

## Results — keys-free baseline (degraded keyword-only mode)

No `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` was set, so this run measures the
single-pass keyword fallback, NOT the agentic search. It is the floor the
agentic mode should beat.

| Query | recall@5 | recall@10 | recall@25 | recall@50 | MRR | time |
|---|---|---|---|---|---|---|
| US5946647A | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 3.9 s |
| US8046721B2 | 0.00 | 0.00 | 0.50 | 0.50 | 0.091 | 11.0 s |
| **aggregate** | **0.50** | **0.50** | **0.75** | **0.75** | **0.545** | — |

Read: the '647 query is easy (distinctive claim vocabulary → both refs found
immediately); slide-to-unlock is hard for naive keywords (generic vocabulary —
"touch", "display", "gesture" — buries the refs; US5821933A missed entirely at
k=50). This is exactly the gap iterative agentic querying (terminology
variants, CPC pivots, narrowing) is designed to close.

## Re-running

```bash
kubectl -n elastic-eval port-forward svc/elasticsearch 9200:9200 &
export ELASTICSEARCH_HOST=http://localhost:9200 ES_USER=elastic ES_PASSWORD=...
export ANTHROPIC_API_KEY=...   # enables the full agentic search + saved traces
.venv/bin/python scripts/run_invalidity_eval.py
```

With a model key the same command runs the agentic loop per query, saves the
reasoning trace to `data/eval_corpus/trace_<patent>.json`, and writes
`eval_report.json` with the same metrics for direct comparison.

## Infra teardown

```bash
gcloud container clusters delete patentkit-eval --project careful-ai --zone us-central1-a
```
