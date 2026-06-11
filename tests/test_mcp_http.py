"""Offline tests for the streamable-http MCP transport (no network, no keys).

Skips cleanly when the ``mcp-http`` extra isn't installed; the stdio
transport-selection tests below the skip guard only need the core package.
"""

from __future__ import annotations

import pytest

from patentkit.integrations.mcp_server import _select_transport

# ---------------------------------------------------------------- transport selection
# (core-only — must run even without the mcp/starlette extras)


def test_default_transport_is_stdio():
    assert _select_transport([], {}) == ("stdio", None)


def test_unrelated_args_and_env_keep_stdio():
    assert _select_transport(["--verbose"], {"PATENTKIT_MCP_TRANSPORT": "stdio"}) == ("stdio", None)


def test_http_flag_without_port():
    assert _select_transport(["--http"], {}) == ("http", None)


def test_http_flag_with_port():
    assert _select_transport(["--http", "9000"], {}) == ("http", 9000)


def test_http_env_selects_http():
    assert _select_transport([], {"PATENTKIT_MCP_TRANSPORT": "HTTP"}) == ("http", None)


def test_cli_flag_wins_over_env():
    assert _select_transport(["--http", "8123"], {"PATENTKIT_MCP_TRANSPORT": "stdio"}) == ("http", 8123)


# ---------------------------------------------------------------- http app + auth

pytest.importorskip("mcp.server.streamable_http_manager", reason="requires the mcp-http extra")
starlette_testclient = pytest.importorskip("starlette.testclient", reason="requires the mcp-http extra")
TestClient = starlette_testclient.TestClient

from patentkit.integrations.mcp_http import BearerAuthMiddleware, build_http_app  # noqa: E402
from patentkit.integrations.toolset import PatentToolset  # noqa: E402

TOKEN = "s3cret-token"

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "patentkit-tests", "version": "0"},
    },
}


async def _ok_app(scope, receive, send):
    """Minimal ASGI app: 200 'ok' for HTTP, no-op completion for lifespan."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_auth_middleware_rejects_missing_and_bad_tokens():
    client = TestClient(BearerAuthMiddleware(_ok_app, TOKEN))
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": f"Bearer {TOKEN}x"}).status_code == 401
    assert client.get("/", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert client.get("/", params={"token": "wrong"}).status_code == 401


def test_auth_middleware_accepts_bearer_header_and_query_param():
    client = TestClient(BearerAuthMiddleware(_ok_app, TOKEN))
    ok = client.get("/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert (ok.status_code, ok.text) == (200, "ok")
    assert client.get("/", params={"token": TOKEN}).status_code == 200


def test_auth_middleware_401_sets_www_authenticate():
    response = TestClient(BearerAuthMiddleware(_ok_app, TOKEN)).get("/")
    assert response.headers["www-authenticate"] == "Bearer"


def test_auth_middleware_passes_lifespan_through():
    # TestClient's context manager fails loudly if lifespan never completes.
    with TestClient(BearerAuthMiddleware(_ok_app, TOKEN)):
        pass


def test_http_app_serves_mcp_initialize():
    app = build_http_app(PatentToolset(), json_response=True)
    with TestClient(app) as client:
        response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["serverInfo"]["name"] == "patentkit"
        assert "tools" in result["capabilities"]


def test_http_app_lists_patentkit_tools():
    from patentkit.integrations.toolset import TOOL_SPECS

    app = build_http_app(PatentToolset(), json_response=True)
    with TestClient(app) as client:
        init = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        session = {"mcp-session-id": init.headers["mcp-session-id"]}
        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={**MCP_HEADERS, **session},
        )
        listed = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={**MCP_HEADERS, **session},
        )
        names = {tool["name"] for tool in listed.json()["result"]["tools"]}
        assert names == {spec["name"] for spec in TOOL_SPECS}


def test_http_app_with_token_requires_auth_end_to_end():
    app = build_http_app(PatentToolset(), token=TOKEN, json_response=True)
    with TestClient(app) as client:
        denied = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert denied.status_code == 401
        allowed = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**MCP_HEADERS, "Authorization": f"Bearer {TOKEN}"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["result"]["serverInfo"]["name"] == "patentkit"


def test_root_path_mirrors_mcp_endpoint():
    app = build_http_app(PatentToolset(), json_response=True)
    with TestClient(app) as client:
        response = client.post("/", json=INITIALIZE, headers=MCP_HEADERS)
        assert response.status_code == 200
        assert response.json()["result"]["serverInfo"]["name"] == "patentkit"
