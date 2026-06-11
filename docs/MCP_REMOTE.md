# Remote MCP: connect claude.ai to patentkit

`patentkit-mcp` defaults to stdio (Claude Desktop). claude.ai custom
connectors instead speak the MCP **streamable-http** transport at a public
HTTPS URL. The recipe: run the server in HTTP mode with a token, tunnel it
with cloudflared, paste the URL into claude.ai.

## 1. Start the server in HTTP mode

```bash
pip install 'patentkit[mcp-http]'   # mcp>=1.8 + starlette + uvicorn

export PATENTKIT_MCP_TOKEN="$(openssl rand -hex 24)"   # ALWAYS set this
echo "token: $PATENTKIT_MCP_TOKEN"
patentkit-mcp --http 8765
# equivalently: PATENTKIT_MCP_TRANSPORT=http patentkit-mcp
```

Startup logs one line confirming transport, bind address, and auth:

```
patentkit-mcp: streamable-http transport listening on http://127.0.0.1:8765/mcp (bearer auth: ON)
```

Knobs: `PATENTKIT_MCP_HOST` (default `127.0.0.1` — the tunnel connects
locally, so don't widen this), `PATENTKIT_MCP_PORT` (default 8765, or pass
`--http <port>`), plus the usual `PATENTKIT_PROVIDER` /
`PATENTKIT_INDEX_JSONL` / `PATENTKIT_SESSION_DIR` from the stdio server.

## 2. Expose it with a cloudflared quick tunnel (no account needed)

```bash
cloudflared tunnel --url http://localhost:8765
```

cloudflared prints a `https://<random>.trycloudflare.com` URL. Quick-tunnel
URLs change on every restart; for a stable hostname use a named tunnel.

## 3. Add the connector in claude.ai

Settings → Connectors → **Add custom connector**, then:

- **URL:** `https://<random>.trycloudflare.com/mcp`
- **Auth:** supply the token as a bearer token if your connector
  configuration offers an OAuth/manual-header field. If the UI gives you no
  way to send a header, fall back to embedding it in the URL:
  `https://<random>.trycloudflare.com/mcp?token=<token>`

Trade-off: the `?token=` query-param fallback works everywhere but the token
then appears in the URL — visible in the connector settings, and routinely
captured in proxy/tunnel access logs. Prefer the `Authorization: Bearer`
header whenever the client can send one, and rotate the token after using the
query-param form.

## Security notes

- The trycloudflare URL is **public**: anyone who finds it can reach your
  server. Always set `PATENTKIT_MCP_TOKEN` — without it the server answers
  unauthenticated requests (it logs `bearer auth: OFF` to warn you).
- The tools are not read-only: they execute searches against your configured
  backends and make LLM calls **billed to your API keys**
  (`PATENTKIT_PROVIDER` credentials). Treat the token like an API key.
- Keep the bind on `127.0.0.1` and let cloudflared do the exposure; kill the
  tunnel when you're done (the URL dies with it).

## Persistent deployment (GCP)

The cloudflared recipe above is the **quick local demo** — the URL dies with
your laptop. For a server that's always reachable and keeps its corpus,
sessions, and graph stores across restarts, deploy to Google Cloud Run with a
GCS bucket mounted at `/data` and secrets in Secret Manager. One scale-to-zero
instance: roughly $0 while idle; LLM calls bill to your key.

```bash
git clone https://github.com/j-hedtke/patentkit && cd patentkit
PROJECT=<your-gcp-project> ./scripts/deploy_gcp.sh
# paste the printed https://...run.app/mcp URL into
# claude.ai -> Settings -> Connectors -> Add custom connector
```

The script is idempotent (re-run it after any failure) and prints the
connector URL and a verification command at the end. For a guided,
conversational path — preflight checks, key handling, common GCP failure
fixes, teardown — use the `deploy-gcp` skill from Claude Code.

### Authentication modes (`AUTH_MODE`)

The deploy defaults to **`oauth-secret`** — the right balance of secure and
frictionless, and it needs no external accounts:

- **`oauth-secret`** (default) — claude.ai performs a real OAuth handshake
  (Dynamic Client Registration + authorization-code + PKCE) against the server;
  you approve once by typing a generated **access code** on a page the server
  hosts, after which a short-lived JWT rides the `Authorization` header. The
  secret never appears in the connector URL. Leave the connector's OAuth
  Client ID/Secret blank — discovery is automatic. Zero extra setup.
- **`oauth-google`** (opt-in: `AUTH_MODE=oauth-google`) — Google sign-in gated
  by an email allowlist (`ALLOWED_EMAILS`), for per-person identity. Costs each
  deployer a one-time Google OAuth web client created in the Cloud console
  (gcloud can't); redirect URI `<service-url>/auth/google/callback`. See
  `src/patentkit/integrations/mcp_oauth.py`.
- **`token`** (opt-in: `AUTH_MODE=token`) — a static bearer secret carried in
  the connector URL. Simplest, least hygienic (token lands in logs/history);
  fine for a throwaway demo.
