---
name: deploy-gcp
description: "Deploy the patentkit MCP server + persistent stores to GCP (Cloud Run + GCS + Secret Manager) through a guided conversation. Use when the user wants a durable/hosted patentkit server, says 'deploy to GCP/cloud', or their tunnel/laptop setup is too fragile."
---

# Deploy patentkit to GCP (guided)

You are walking a user — assume a patent attorney with zero infra experience —
through deploying a persistent patentkit MCP server. The heavy lifting is in
`scripts/deploy_gcp.sh` (idempotent; safe to re-run after any failure). Your
job is the conversation around it: preflight, collecting inputs, watching the
run, and handing over a working claude.ai connector. Explain each step in one
plain sentence before running it; never paste raw gcloud errors at the user
without a translation.

What gets created (so you can answer "what is this going to do?"):
one Cloud Run service (`patentkit-mcp --http`, max 1 instance, scales to
zero), one GCS bucket mounted at `/data` for durable corpus/sessions/graph
files, two Secret Manager secrets (their Anthropic key + an auto-generated
access token), and one container image in Artifact Registry.

## 1. Preflight

- `gcloud` installed? `command -v gcloud`. If missing, point them at
  https://cloud.google.com/sdk/docs/install (on macOS, `brew install
  google-cloud-sdk` is simplest), then `gcloud auth login`.
- Authenticated? `gcloud auth list` — there must be an ACTIVE account. If
  not: `gcloud auth login` (opens a browser; they click through).
- Pick the project: run `gcloud projects list` and ASK the user which
  project to use — do not guess. If they have none, walk them through
  `gcloud projects create` plus attaching billing in the console
  (console.cloud.google.com/billing). Confirm billing is enabled:
  `gcloud billing projects describe <PROJECT>` should show
  `billingEnabled: true` (if the billing API isn't enabled, just ask them to
  check the Billing page in the console).
- Region: default `us-central1`; only ask if they care about data residency
  or latency.

## 2. Collect the Anthropic API key

The server makes LLM calls billed to the user's own Anthropic key.

1. First check whether a key is already available — `ANTHROPIC_API_KEY` in
   the environment or in a local `.env`. If found, confirm with the user
   that it's the one to use and don't ask them to paste anything.
2. If not available, ask them to paste it — with this one-line warning:
   pasting it into this chat means it lives in the chat transcript, and the
   alternative is to run the storing command themselves. The alternative:
   after the secret exists, they run
   `gcloud secrets versions add patentkit-anthropic-key --data-file=-`
   in their own terminal (type the key, then Ctrl-D) and you skip handling
   it entirely (the script only creates the secret if missing, so create it
   with a placeholder is NOT okay — instead let the script prompt them
   interactively, or have them export `ANTHROPIC_API_KEY` in the shell that
   runs the script).
3. If they do paste the key: put it into Secret Manager (or the script's
   env) immediately, never echo it back, never include it in any later
   message or command output you show them.

## 3. Run the deployment

```bash
PROJECT=<their-project> REGION=<region> ./scripts/deploy_gcp.sh
```

(Export `ANTHROPIC_API_KEY` in the same invocation if you collected it.)
The script is idempotent — on any failure, fix the cause and re-run the same
command; completed steps are skipped. Expect the Cloud Build step to take a
few minutes the first time.

Common failures and fixes:

- **Billing disabled** — errors mentioning `billing` or
  `FAILED_PRECONDITION` when enabling APIs. Fix: enable billing for the
  project at console.cloud.google.com/billing, then re-run.
- **Org policy blocks unauthenticated access** — `gcloud run deploy
  --allow-unauthenticated` fails with an IAM/org-policy error
  (`constraints/iam.allowedPolicyMemberDomains` or "setIamPolicy permission
  denied" on `allUsers`). This happens on company-managed GCP orgs. Options:
  ask their GCP admin to exempt the project, or deploy without
  `--allow-unauthenticated` and accept that claude.ai connectors can't reach
  it directly (they'd need an authenticated proxy — flag this honestly
  rather than working around it silently). The bearer token is the real
  auth either way.
- **Cloud Build first-run grants** — the very first `gcloud builds submit`
  in a project can fail with permission errors while the Cloud Build service
  account is being provisioned. Wait ~1 minute and re-run; if it persists,
  grant the Cloud Build SA (`<project-number>@cloudbuild.gserviceaccount.com`)
  the Artifact Registry Writer role.
- **`gsutil` not found** — it ships with the gcloud SDK; `gcloud components
  install gsutil` or reinstall the SDK.

## 4. Post-deploy: verify and hand over

1. Run the curl `initialize` probe the script printed (POST to
   `<service-url>/mcp` with the bearer token). A JSON-RPC `result` with
   `serverInfo` = success; a 401 means the token is wrong; a 404/503 means
   the service isn't up — check `gcloud run services logs read <service>`.
2. claude.ai connector: Settings → Connectors → **Add custom connector** →
   paste `<service-url>/mcp?token=<token>` (the script printed the exact
   URL). Mention once: the `?token=` form puts the token in the URL; if
   their client can send an `Authorization: Bearer` header, prefer that and
   use the bare `/mcp` URL.
3. Claude Desktop alternative (its connector UI also accepts remote MCP
   servers, or via `mcpServers` config with a streamable-http client such
   as `mcp-remote`):
   ```json
   {
     "mcpServers": {
       "patentkit": {
         "command": "npx",
         "args": ["-y", "mcp-remote", "<service-url>/mcp",
                  "--header", "Authorization: Bearer <token>"]
       }
     }
   }
   ```
4. Costs, stated plainly: the service scales to zero, so idle cost is
   approximately $0 (pennies for storage); every search/analysis makes LLM
   calls billed to their Anthropic key. Teardown when done:
   `gcloud run services delete <service> --region <region>`, delete the
   bucket (`gsutil rm -r gs://<bucket>`) and the two secrets
   (`gcloud secrets delete patentkit-anthropic-key` / `patentkit-mcp-token`).

## 5. Corpus guidance

The server preloads `/data/corpus.jsonl` from the bucket at startup. If
`data/eval_corpus/corpus.jsonl` didn't exist locally at deploy time, the
bucket is empty and searches will run over an empty index. Two options:

- Build a corpus locally with `scripts/build_eval_corpus.py`, then
  `gsutil cp data/eval_corpus/corpus.jsonl gs://<bucket>/corpus.jsonl` and
  redeploy/restart (or just re-run the deploy script — it uploads if the
  bucket lacks one).
- Or index patents post-deploy through the server itself: the `index_patents`
  MCP tool accepts patent numbers and fetches them live (writes land on the
  mounted bucket, so they persist).
