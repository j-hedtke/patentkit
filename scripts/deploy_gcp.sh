#!/usr/bin/env bash
# Deploy the patentkit MCP server to Google Cloud Run with durable storage.
#
# What you get:
#   * a Cloud Run service running `patentkit-mcp --http` (scales to zero when
#     idle — roughly $0 when nobody is using it)
#   * a GCS bucket mounted at /data inside the container, so the corpus,
#     guided-search sessions, and graph stores survive restarts
#   * your Anthropic API key and an auto-generated access token stored in
#     Secret Manager (never baked into the image or printed to logs — the
#     token IS echoed once at the end so you can paste it into claude.ai)
#
# Usage:
#   PROJECT=my-gcp-project ./scripts/deploy_gcp.sh
#   ./scripts/deploy_gcp.sh --project my-gcp-project [--region us-central1] \
#       [--service patentkit-mcp] [--bucket my-bucket]
#
# Safe to re-run: every step checks whether it already happened.

set -euo pipefail

usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- parameters
# Flags override environment variables; environment variables override
# defaults. Only PROJECT is required.
PROJECT="${PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-patentkit-mcp}"
BUCKET="${BUCKET:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region)  REGION="$2";  shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --bucket)  BUCKET="$2";  shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: a GCP project is required: PROJECT=<id> or --project <id>." >&2
  echo "       List yours with: gcloud projects list" >&2
  exit 1
fi
BUCKET="${BUCKET:-${PROJECT}-patentkit-data}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR_REPO="patentkit"                                    # Artifact Registry repo
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/patentkit-mcp:latest"
KEY_SECRET="patentkit-anthropic-key"                   # Secret Manager names
TOKEN_SECRET="patentkit-mcp-token"
CORPUS_LOCAL="${REPO_ROOT}/data/eval_corpus/corpus.jsonl"

echo "==> Deploying patentkit MCP server"
echo "    project: ${PROJECT}   region: ${REGION}"
echo "    service: ${SERVICE}   bucket: gs://${BUCKET}"
echo

# ------------------------------------------------------------- 1. enable APIs
# Turn on the four Google Cloud products this deployment uses. Re-running is
# a no-op when they are already enabled.
echo "==> [1/8] Enabling required Google Cloud APIs (no-op if already on)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT"

# -------------------------------------------------- 2. Artifact Registry repo
# A private place to keep the built container image.
echo "==> [2/8] Ensuring Artifact Registry repo '${AR_REPO}' exists..."
if ! gcloud artifacts repositories describe "$AR_REPO" \
    --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format docker --location "$REGION" --project "$PROJECT" \
    --description "patentkit container images"
else
  echo "    already exists — skipping."
fi

# --------------------------------------------------------------- 3. GCS bucket
# Durable storage. Cloud Run mounts this bucket at /data inside the
# container, so the corpus / sessions / graph files survive restarts.
echo "==> [3/8] Ensuring storage bucket gs://${BUCKET} exists..."
if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
  gsutil mb -p "$PROJECT" -l "$REGION" "gs://${BUCKET}"
else
  echo "    already exists — skipping."
fi

# ------------------------------------------------------------ 4. seed corpus
# If you have a local corpus and the bucket doesn't have one yet, upload it
# so the server preloads it at startup (PATENTKIT_INDEX_JSONL). Skipped when
# there's no local corpus (you can index patents post-deploy with the
# index_patents tool, or build one with scripts/build_eval_corpus.py).
echo "==> [4/8] Seeding the corpus into the bucket (if available)..."
if [[ -f "$CORPUS_LOCAL" ]]; then
  if gsutil -q stat "gs://${BUCKET}/corpus.jsonl"; then
    echo "    bucket already has corpus.jsonl — not overwriting."
  else
    gsutil cp "$CORPUS_LOCAL" "gs://${BUCKET}/corpus.jsonl"
  fi
else
  echo "    no local ${CORPUS_LOCAL#"$REPO_ROOT"/} — skipping (index patents post-deploy)."
fi

# ---------------------------------------------------------------- 5. secrets
# The Anthropic API key (yours) and the server access token (auto-generated)
# live in Secret Manager and are injected into the container as env vars.
# Existing secrets are left untouched on re-runs.
echo "==> [5/8] Ensuring secrets exist in Secret Manager..."
if ! gcloud secrets describe "$KEY_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    read -rs -p "    Paste your Anthropic API key (input hidden): " ANTHROPIC_API_KEY
    echo
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: no Anthropic API key provided (env ANTHROPIC_API_KEY or prompt)." >&2
    exit 1
  fi
  printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets create "$KEY_SECRET" \
    --project "$PROJECT" --replication-policy automatic --data-file=-
  echo "    created secret ${KEY_SECRET}."
else
  echo "    secret ${KEY_SECRET} already exists — leaving it as-is."
  echo "    (rotate with: gcloud secrets versions add ${KEY_SECRET} --data-file=- )"
fi

if ! gcloud secrets describe "$TOKEN_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  MCP_TOKEN="$(openssl rand -hex 32)"
  printf '%s' "$MCP_TOKEN" | gcloud secrets create "$TOKEN_SECRET" \
    --project "$PROJECT" --replication-policy automatic --data-file=-
  echo "    generated and stored a fresh access token (${TOKEN_SECRET})."
else
  MCP_TOKEN="$(gcloud secrets versions access latest \
    --secret "$TOKEN_SECRET" --project "$PROJECT")"
  echo "    secret ${TOKEN_SECRET} already exists — reusing it."
fi

# ------------------------------------------------------------- 6. permissions
# Let the Cloud Run runtime identity (the default compute service account)
# read the two secrets and read/write objects in the bucket. These grants
# are idempotent.
echo "==> [6/8] Granting the Cloud Run service account access..."
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for secret in "$KEY_SECRET" "$TOKEN_SECRET"; do
  gcloud secrets add-iam-policy-binding "$secret" --project "$PROJECT" \
    --member "serviceAccount:${RUNTIME_SA}" \
    --role roles/secretmanager.secretAccessor >/dev/null
done
gsutil iam ch "serviceAccount:${RUNTIME_SA}:roles/storage.objectAdmin" "gs://${BUCKET}"

# ------------------------------------------------------------ 7. build image
# Cloud Build builds the Dockerfile in this repo and pushes the image to
# Artifact Registry — no local Docker required.
echo "==> [7/8] Building and pushing the container image (Cloud Build)..."
gcloud builds submit "$REPO_ROOT" --tag "$IMAGE" --project "$PROJECT"

# ---------------------------------------------------------------- 8. deploy
# Single instance max: the stores are plain files on the mounted bucket and
# are not safe for concurrent writers. Min instances 0 = scales to zero =
# no compute cost while idle. Auth is the bearer token (that's why
# --allow-unauthenticated is safe here). gen2 execution environment is
# required for Cloud Storage volume mounts. The long request timeout covers
# multi-minute agentic searches.
echo "==> [8/8] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 3600 \
  --execution-environment gen2 \
  --add-volume "name=data,type=cloud-storage,bucket=${BUCKET}" \
  --add-volume-mount "volume=data,mount-path=/data" \
  --set-env-vars "PATENTKIT_PROVIDER=anthropic,PATENTKIT_INDEX_JSONL=/data/corpus.jsonl,PATENTKIT_SESSION_DIR=/data/sessions" \
  --set-secrets "ANTHROPIC_API_KEY=${KEY_SECRET}:latest,PATENTKIT_MCP_TOKEN=${TOKEN_SECRET}:latest"

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" --format 'value(status.url)')"

# ----------------------------------------------------------------- summary
cat <<EOF

============================================================
patentkit MCP server deployed.

Service URL:    ${SERVICE_URL}
MCP endpoint:   ${SERVICE_URL}/mcp
Access token:   ${MCP_TOKEN}

claude.ai connector URL (Settings -> Connectors -> Add custom connector):

  ${SERVICE_URL}/mcp?token=${MCP_TOKEN}

(Prefer sending the token as an "Authorization: Bearer" header if your
client supports it; the ?token= form works everywhere but shows up in URLs.)

Verify the server answers (expects an "initialize" result, not a 401):

  curl -sS "${SERVICE_URL}/mcp" -X POST \\
    -H "Authorization: Bearer ${MCP_TOKEN}" \\
    -H "Content-Type: application/json" \\
    -H "Accept: application/json, text/event-stream" \\
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'

Costs: the service scales to zero when idle (~\$0); LLM calls bill to your
Anthropic key. Tear down with:
  gcloud run services delete ${SERVICE} --region ${REGION} --project ${PROJECT}
  gsutil rm -r gs://${BUCKET}
  gcloud secrets delete ${KEY_SECRET} --project ${PROJECT}
  gcloud secrets delete ${TOKEN_SECRET} --project ${PROJECT}
============================================================
EOF
