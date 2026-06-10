#!/usr/bin/env bash
# End-to-end IPR-example invalidity eval, from a fresh clone:
#   1. venv + extras
#   2. Elasticsearch (local Docker by default; or point ELASTICSEARCH_HOST at
#      an existing cluster — see .claude/skills/eval-e2e/SKILL.md for the
#      managed-GKE variant)
#   3. corpus (live Google Patents scrape, resumable, reused if present)
#   4. index into ES
#   5. eval twice: keys-free baseline, then agentic (needs ANTHROPIC_API_KEY
#      or OPENAI_API_KEY; the baseline alone is NOT agentic performance)
#
# Usage: scripts/eval_e2e.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
ES_CONTAINER=patentkit-eval-es
ES_VERSION=8.14.3
OUT=data/eval_corpus

if [ ! -x "$PY" ]; then
  echo "creating .venv and installing patentkit extras..."
  python3 -m venv .venv
fi
"$PY" -c "import anthropic, openai, elasticsearch, bs4" 2>/dev/null ||
  .venv/bin/pip install -q -e '.[anthropic,openai,elasticsearch,scrape]'

# --- Elasticsearch ----------------------------------------------------------
if [ -z "${ELASTICSEARCH_HOST:-}" ]; then
  command -v docker >/dev/null ||
    { echo "docker not found; install it or set ELASTICSEARCH_HOST to an existing cluster" >&2; exit 1; }
  if ! docker ps --format '{{.Names}}' | grep -qx "$ES_CONTAINER"; then
    docker rm -f "$ES_CONTAINER" >/dev/null 2>&1 || true
    echo "starting local Elasticsearch $ES_VERSION (container $ES_CONTAINER)..."
    docker run -d --name "$ES_CONTAINER" -p 9200:9200 \
      -e discovery.type=single-node -e xpack.security.enabled=false \
      -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
      "docker.elastic.co/elasticsearch/elasticsearch:$ES_VERSION" >/dev/null
  fi
  export ELASTICSEARCH_HOST=http://localhost:9200
  export ES_USER=elastic ES_PASSWORD=unused  # security disabled locally; client requires the vars
fi

probe() {
  if [ -n "${ELASTICSEARCH_API_KEY:-}" ]; then
    curl -fsSk -H "Authorization: ApiKey $ELASTICSEARCH_API_KEY" "$ELASTICSEARCH_HOST" >/dev/null 2>&1
  else
    curl -fsSk -u "${ES_USER:-elastic}:${ES_PASSWORD:-unused}" "$ELASTICSEARCH_HOST" >/dev/null 2>&1
  fi
}
echo "waiting for Elasticsearch at $ELASTICSEARCH_HOST..."
ok=
for _ in $(seq 1 60); do probe && { ok=1; break; }; sleep 2; done
[ -n "$ok" ] || { echo "Elasticsearch did not become ready" >&2; exit 1; }

# --- Corpus ------------------------------------------------------------------
if [ ! -s "$OUT/corpus.jsonl" ]; then
  echo "building eval corpus (live scrape of ~300 patents, ~15 min; resumable on rerun)..."
  "$PY" scripts/build_eval_corpus.py --out "$OUT"
fi

"$PY" scripts/index_eval_corpus.py --corpus "$OUT/corpus.jsonl"

# --- Eval --------------------------------------------------------------------
echo
echo "=== keys-free baseline (degraded keyword-only mode — NOT agentic performance) ==="
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY "$PY" scripts/run_invalidity_eval.py --out "$OUT"
mv "$OUT/eval_report.json" "$OUT/eval_report.keysfree-baseline.json"

if [ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  echo
  echo "=== agentic run ==="
  "$PY" scripts/run_invalidity_eval.py --out "$OUT"
  echo
  echo "baseline: $OUT/eval_report.keysfree-baseline.json"
  echo "agentic:  $OUT/eval_report.json (reasoning traces: $OUT/trace_*.json)"
else
  echo
  echo "No ANTHROPIC_API_KEY/OPENAI_API_KEY set — agentic run skipped."
  echo "The numbers above are the degraded fallback floor, not agentic performance."
fi
