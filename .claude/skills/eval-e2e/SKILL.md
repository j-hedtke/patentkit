---
name: eval-e2e
description: Set up the toy IPR eval from scratch (Elasticsearch, scraped corpus, index) and run the invalidity-search experiments — keys-free baseline AND live agentic — then write up the comparison. Use when asked to "run the eval", "reproduce the IPR experiments", or "set up the eval cluster".
---

# End-to-end IPR invalidity eval

Goal: from any checkout state, produce a baseline-vs-agentic comparison of
`InvaliditySearchAgent` on the two IPR-example query patents (US5946647A,
US8046721B2) over a ~300-patent real scraped corpus, and report both numbers
**clearly labeled by mode**.

The non-interactive path is `scripts/eval_e2e.sh` — prefer running it and
supervising, stepping in only where it fails. The steps below are what it does,
plus the judgment calls it can't make.

## 0. Prerequisites

- Python ≥3.11 venv at `.venv` with extras: `pip install -e '.[anthropic,openai,elasticsearch,scrape]'`.
  A missing `anthropic`/`openai` SDK does NOT abort the eval — the tool loop
  catches the ImportError per step and silently produces a zeroed "agentic"
  report. Verify the import works before trusting any agentic numbers.
- API keys: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (repo convention: `.env` at
  the repo root, loaded by direnv via `.envrc`; the Bash tool shell may not run
  the direnv hook — `set -a; source .env; set +a` explicitly).
- Without keys you can still run the baseline, but it MUST be reported as
  "degraded keyword-only mode", never as agentic performance.

## 1. Elasticsearch

Pick one:

**A. Local Docker (default — free, simplest):**
```sh
docker run -d --name patentkit-eval-es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.14.3
export ELASTICSEARCH_HOST=http://localhost:9200 ES_USER=elastic ES_PASSWORD=unused
```

**B. Existing cluster:** set `ELASTICSEARCH_HOST` + either
`ELASTICSEARCH_API_KEY` or `ES_USER`/`ES_PASSWORD` (that precedence is
hardcoded in `scripts/index_eval_corpus.py:build_client`).

**C. Managed GKE (shared/billable — only if explicitly requested):**
```sh
gcloud container clusters create patentkit-eval --project careful-ai \
  --zone us-central1-a --num-nodes 1 --machine-type e2-standard-4
# deploy single-node ES 8.14.3 (security on, TLS off) into namespace
# elastic-eval as deployment+service "elasticsearch", then:
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
kubectl -n elastic-eval port-forward svc/elasticsearch 9200:9200 &
export ELASTICSEARCH_HOST=http://localhost:9200 ES_USER=elastic
export ES_PASSWORD="$(cat data/eval_corpus/.es_password)"
```
⚠ ~$0.33/hr until deleted. Always surface the running cost and offer teardown
when the experiments are done:
`gcloud container clusters delete patentkit-eval --project careful-ai --zone us-central1-a`.

Gotchas:
- ES client pin is `<9` (v9 client speaks a media type 8.x servers reject).
- "address already in use" on 9200 usually means a stale port-forward or an
  old container is still serving — check what's there before assuming failure;
  it may already be the cluster you want.

## 2. Corpus

Reuse `data/eval_corpus/corpus.jsonl` if present (~120MB, gitignored).
Otherwise `python scripts/build_eval_corpus.py` — a **live Google Patents
scrape** (~300 patents, ~15 min, resumable; reruns skip already-scraped
records). Never substitute synthetic fixtures; real data is a project
invariant. The manifest (`manifest.json`) carries the querysets and
ground-truth references.

## 3. Index

```sh
python scripts/index_eval_corpus.py            # add --recreate to rebuild
```
Sanity-check: `GET $ELASTICSEARCH_HOST/patentkit-eval-corpus/_count` → 300.
(Field names if hand-querying: `number` / `patent_number_norm`; the mapping is
`dynamic: false`, so unknown fields silently aggregate to nothing.)

## 4. Run both experiments

Both runs share `--out data/eval_corpus` (the script reads `manifest.json`
from `--out`, so don't split output dirs). Baseline first, then rename its
report so the agentic run doesn't overwrite it:

```sh
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY \
  python scripts/run_invalidity_eval.py
mv data/eval_corpus/eval_report.json data/eval_corpus/eval_report.keysfree-baseline.json

python scripts/run_invalidity_eval.py          # agentic (keys in env)
```

The agentic run takes minutes per query (tool-use loop, 180s/16-step budget;
runs in the background if long). Confirm the log opens with `LLM: anthropic`
(or `openai`) and shows real `agent -> search_patents` rounds — see
prerequisite note about silently-zeroed reports. Reasoning traces land in
`data/eval_corpus/trace_<patent>.json`.

## 5. Report

Write the comparison to `docs/evals/<YYYY-MM-DD>-<slug>.md` (follow
`docs/evals/2026-06-10-ipr-es-eval.md` as the template): setup, per-query and
aggregate recall@k/MRR for **both modes side by side**, elapsed times,
`stop_reason`s, and which ground-truth references were found/missed. Rules:

- Label degraded results as degraded, everywhere, every time.
- `stop_reason: "budget_exceeded"` means the agent never chose to stop —
  flag it; the numbers may understate the agent.
- Note the LLM provider/model and `final_k` so runs are comparable.
- If using GKE, end by reminding about the billable cluster.
