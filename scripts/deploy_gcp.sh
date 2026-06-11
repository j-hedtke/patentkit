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
SIGNING_SECRET="patentkit-oauth-signing"               # HS256 key for issued JWTs
GOOGLE_SECRET="patentkit-google-oauth-secret"          # Google OAuth client secret
CORPUS_LOCAL="${REPO_ROOT}/data/eval_corpus/corpus.jsonl"

# AUTH_MODE (default "oauth-secret"):
#   oauth-secret  — claude.ai authenticates via OAuth; you approve once by
#                   typing a generated access code on a local page, then a
#                   short-lived token rides the Authorization header. No
#                   external identity provider, no extra setup. RECOMMENDED.
#   oauth-google  — Google sign-in gated by an email allowlist. Per-person
#                   identity, but each deployer must create a Google OAuth web
#                   client in the console (gcloud can't) and set, via env/flags:
#                   GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET,
#                   ALLOWED_EMAILS. Redirect URI: <service-url>/auth/google/callback
#   token         — static bearer secret carried in the connector URL. Simplest,
#                   least hygienic (token in logs/history); fine for quick demos.
AUTH_MODE="${AUTH_MODE:-oauth-secret}"

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

# A generated secret in $TOKEN_SECRET serves as the static bearer (token mode)
# OR the access code typed on the approve page (oauth-secret mode). Created
# once, reused on reruns; surfaced as MCP_TOKEN for the summary.
ensure_token_secret() {
  if ! gcloud secrets describe "$TOKEN_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
    MCP_TOKEN="$(openssl rand -hex 32)"
    printf '%s' "$MCP_TOKEN" | gcloud secrets create "$TOKEN_SECRET" \
      --project "$PROJECT" --replication-policy automatic --data-file=-
    echo "    generated and stored the access secret (${TOKEN_SECRET})."
  else
    MCP_TOKEN="$(gcloud secrets versions access latest \
      --secret "$TOKEN_SECRET" --project "$PROJECT")"
    echo "    secret ${TOKEN_SECRET} already exists — reusing it."
  fi
}

# HS256 key patentkit uses to sign the JWTs it issues; generate once, reuse so
# live sessions survive redeploys.
ensure_signing_secret() {
  if ! gcloud secrets describe "$SIGNING_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
    openssl rand -hex 32 | gcloud secrets create "$SIGNING_SECRET" \
      --project "$PROJECT" --replication-policy automatic --data-file=-
    echo "    generated token-signing key (${SIGNING_SECRET})."
  else
    echo "    signing key ${SIGNING_SECRET} already exists — reusing it."
  fi
}

if [[ "$AUTH_MODE" == "oauth-google" ]]; then
  if [[ -z "${GOOGLE_OAUTH_CLIENT_ID:-}" || -z "${GOOGLE_OAUTH_CLIENT_SECRET:-}" || -z "${ALLOWED_EMAILS:-}" ]]; then
    echo "ERROR: oauth-google mode needs GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET," >&2
    echo "       and ALLOWED_EMAILS (create the OAuth web client in the Google Cloud console;" >&2
    echo "       its authorized redirect URI is <service-url>/auth/google/callback)." >&2
    exit 1
  fi
  if gcloud secrets describe "$GOOGLE_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
    printf '%s' "$GOOGLE_OAUTH_CLIENT_SECRET" | gcloud secrets versions add "$GOOGLE_SECRET" \
      --project "$PROJECT" --data-file=-
  else
    printf '%s' "$GOOGLE_OAUTH_CLIENT_SECRET" | gcloud secrets create "$GOOGLE_SECRET" \
      --project "$PROJECT" --replication-policy automatic --data-file=-
  fi
  echo "    stored Google OAuth client secret (${GOOGLE_SECRET})."
  ensure_signing_secret
elif [[ "$AUTH_MODE" == "oauth-secret" ]]; then
  ensure_token_secret      # the access code typed on the approve page
  ensure_signing_secret
else
  ensure_token_secret      # token mode: static bearer in the URL
fi

# ------------------------------------------------------------- 6. permissions
# Let the Cloud Run runtime identity (the default compute service account)
# read the two secrets and read/write objects in the bucket. These grants
# are idempotent.
echo "==> [6/8] Granting the Cloud Run service account access..."
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
case "$AUTH_MODE" in
  oauth-google) GRANT_SECRETS=("$KEY_SECRET" "$GOOGLE_SECRET" "$SIGNING_SECRET") ;;
  oauth-secret) GRANT_SECRETS=("$KEY_SECRET" "$TOKEN_SECRET" "$SIGNING_SECRET") ;;
  *)            GRANT_SECRETS=("$KEY_SECRET" "$TOKEN_SECRET") ;;
esac
for secret in "${GRANT_SECRETS[@]}"; do
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
# In oauth-google mode the issuer/PUBLIC_URL must equal the service's own URL,
# which only exists after the service is first created. Resolve it up front;
# if the service is brand new, do a token-mode-style first deploy is avoided by
# requiring the service to already exist (run once in token mode, or just let
# this resolve to empty and re-run — Cloud Run assigns a stable URL on create).
PUBLIC_URL="$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" --format 'value(status.url)' 2>/dev/null || true)"

COMMON_ENV="PATENTKIT_PROVIDER=anthropic,PATENTKIT_INDEX_JSONL=/data/corpus.jsonl,PATENTKIT_SESSION_DIR=/data/sessions"
if [[ "$AUTH_MODE" == oauth-* && -z "$PUBLIC_URL" ]]; then
  echo "ERROR: ${AUTH_MODE} needs the service URL up front, but ${SERVICE} does not exist yet." >&2
  echo "       Deploy once in token mode (AUTH_MODE=token) to mint the URL, then re-run." >&2
  exit 1
fi
case "$AUTH_MODE" in
  oauth-google)
    ENV_VARS="${COMMON_ENV},PATENTKIT_AUTH_MODE=oauth-google,PATENTKIT_PUBLIC_URL=${PUBLIC_URL},GOOGLE_OAUTH_CLIENT_ID=${GOOGLE_OAUTH_CLIENT_ID},PATENTKIT_ALLOWED_EMAILS=${ALLOWED_EMAILS},PATENTKIT_OAUTH_DIR=/data/oauth"
    SECRETS="ANTHROPIC_API_KEY=${KEY_SECRET}:latest,GOOGLE_OAUTH_CLIENT_SECRET=${GOOGLE_SECRET}:latest,PATENTKIT_OAUTH_SIGNING_SECRET=${SIGNING_SECRET}:latest"
    ;;
  oauth-secret)
    ENV_VARS="${COMMON_ENV},PATENTKIT_AUTH_MODE=oauth-secret,PATENTKIT_PUBLIC_URL=${PUBLIC_URL},PATENTKIT_OAUTH_DIR=/data/oauth"
    SECRETS="ANTHROPIC_API_KEY=${KEY_SECRET}:latest,PATENTKIT_ACCESS_SECRET=${TOKEN_SECRET}:latest,PATENTKIT_OAUTH_SIGNING_SECRET=${SIGNING_SECRET}:latest"
    ;;
  *)
    ENV_VARS="$COMMON_ENV"
    SECRETS="ANTHROPIC_API_KEY=${KEY_SECRET}:latest,PATENTKIT_MCP_TOKEN=${TOKEN_SECRET}:latest"
    ;;
esac

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
  --set-env-vars "$ENV_VARS" \
  --set-secrets "$SECRETS"

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" --format 'value(status.url)')"

# ----------------------------------------------------------------- summary
echo
echo "============================================================"
echo "patentkit MCP server deployed."
echo
echo "Service URL:    ${SERVICE_URL}"
echo "MCP endpoint:   ${SERVICE_URL}/mcp"
echo
if [[ "$AUTH_MODE" == "oauth-secret" ]]; then
  cat <<EOF
Auth: OAuth (self-contained — no external identity provider).

Access code (you'll type this once on the approve page, then never again):
  ${MCP_TOKEN}

claude.ai connector (Settings -> Connectors -> Add custom connector):
  Name: patentkit
  Remote MCP server URL: ${SERVICE_URL}/mcp
  Leave OAuth Client ID / Secret BLANK — claude.ai discovers the auth server,
  registers itself, then sends you to the approve page to enter the access
  code above. After that, a short-lived token rides the Authorization header.

Verify discovery (expects JSON metadata with authorize/token endpoints):
  curl -sS "${SERVICE_URL}/.well-known/oauth-authorization-server"
EOF
elif [[ "$AUTH_MODE" == "oauth-google" ]]; then
  cat <<EOF
Auth: Google sign-in (allowlist: ${ALLOWED_EMAILS}).

In the Google Cloud console, the OAuth web client's Authorized redirect URI
MUST be exactly:
  ${SERVICE_URL}/auth/google/callback

claude.ai connector (Settings -> Connectors -> Add custom connector):
  Name: patentkit
  Remote MCP server URL: ${SERVICE_URL}/mcp
  Leave OAuth Client ID / Secret BLANK — claude.ai discovers the auth server
  and registers itself; you'll sign in with Google on first use.

Verify discovery (expects JSON metadata with authorize/token endpoints):
  curl -sS "${SERVICE_URL}/.well-known/oauth-authorization-server"
EOF
else
  cat <<EOF
Access token:   ${MCP_TOKEN}

claude.ai connector URL (Settings -> Connectors -> Add custom connector):
  ${SERVICE_URL}/mcp?token=${MCP_TOKEN}

(For stronger auth, redeploy with AUTH_MODE=oauth-google — Google sign-in,
no token in the URL.)

Verify the server answers (expects an "initialize" result, not a 401):
  curl -sS "${SERVICE_URL}/mcp" -X POST \\
    -H "Authorization: Bearer ${MCP_TOKEN}" \\
    -H "Content-Type: application/json" \\
    -H "Accept: application/json, text/event-stream" \\
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'
EOF
fi
cat <<EOF

Costs: the service scales to zero when idle (~\$0); LLM calls bill to your
Anthropic key. Tear down with:
  gcloud run services delete ${SERVICE} --region ${REGION} --project ${PROJECT}
  gsutil rm -r gs://${BUCKET}
  gcloud secrets delete ${KEY_SECRET} --project ${PROJECT}
============================================================
EOF
