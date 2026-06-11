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
