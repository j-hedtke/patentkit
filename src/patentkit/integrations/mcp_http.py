"""Streamable-HTTP transport for the patentkit MCP server.

Serves the same low-level :mod:`mcp` Server built by
:func:`patentkit.integrations.mcp_server.build_server` over the MCP spec's
streamable-http transport (the transport claude.ai custom connectors speak),
using the SDK's :class:`~mcp.server.streamable_http_manager.
StreamableHTTPSessionManager` mounted in a small Starlette app.

Select it on the ``patentkit-mcp`` entrypoint with ``--http [port]`` or
``PATENTKIT_MCP_TRANSPORT=http``; the default transport remains stdio.

Environment configuration (in addition to the variables documented in
:mod:`patentkit.integrations.mcp_server`):

- ``PATENTKIT_MCP_HOST``   bind address (default ``127.0.0.1`` — tunnel locally)
- ``PATENTKIT_MCP_PORT``   port when ``--http`` is given without one (default 8765)
- ``PATENTKIT_MCP_TOKEN``  if set, every HTTP request must carry
  ``Authorization: Bearer <token>`` (or ``?token=<token>``) or it is rejected
  with 401. Always set this before exposing the server through a tunnel.

Requires the ``mcp-http`` extra: ``pip install 'patentkit[mcp-http]'``.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
from typing import Any
from urllib.parse import parse_qs

from patentkit.integrations.mcp_server import build_server, build_toolset

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
MCP_PATH = "/mcp"

_INSTALL_HINT = (
    "patentkit's MCP streamable-http transport requires the 'mcp-http' extra "
    "(mcp>=1.8 with starlette and uvicorn). "
    "Install it with: pip install 'patentkit[mcp-http]'"
)


class BearerAuthMiddleware:
    """ASGI middleware that rejects HTTP requests lacking the shared token.

    Accepts ``Authorization: Bearer <token>`` or, as a fallback for clients
    whose connector UI cannot set headers, ``?token=<token>`` in the query
    string (note: query strings tend to end up in proxy/tunnel logs — prefer
    the header). Non-HTTP scopes (lifespan, websocket) pass through untouched.
    """

    def __init__(self, app: Any, token: str) -> None:
        if not token:
            raise ValueError("BearerAuthMiddleware requires a non-empty token")
        self.app = app
        self._token = token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        body = json.dumps(
            {"error": "unauthorized", "detail": "missing or invalid bearer token"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _authorized(self, scope: Any) -> bool:
        for name, value in scope.get("headers") or ():
            if name == b"authorization":
                scheme, _, credential = value.decode("latin-1").partition(" ")
                if scheme.lower() == "bearer" and hmac.compare_digest(
                    credential.strip(), self._token
                ):
                    return True
        query = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
        return any(hmac.compare_digest(candidate, self._token) for candidate in query.get("token", []))


def build_http_app(
    toolset: Any = None,
    *,
    token: str | None = None,
    json_response: bool = False,
    stateless: bool = False,
) -> Any:
    """Build the ASGI app serving the patentkit MCP server at ``/mcp``.

    The MCP endpoint is also mirrored at ``/`` so a bare tunnel URL works.
    If ``token`` is given the whole app is wrapped in
    :class:`BearerAuthMiddleware`.
    """
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: PLC0415
        from starlette.applications import Starlette  # noqa: PLC0415
        from starlette.routing import Route  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    if toolset is None:
        toolset = build_toolset()
    server = build_server(toolset)
    manager = StreamableHTTPSessionManager(
        app=server, json_response=json_response, stateless=stateless
    )

    class _StreamableHTTPEndpoint:
        """ASGI endpoint delegating to the session manager (Route treats class instances as ASGI)."""

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any):
        async with manager.run():
            yield

    endpoint = _StreamableHTTPEndpoint()
    app: Any = Starlette(
        routes=[Route(MCP_PATH, endpoint=endpoint), Route("/", endpoint=endpoint)],
        lifespan=lifespan,
    )
    if token:
        app = BearerAuthMiddleware(app, token)
    return app


def serve_http(*, host: str | None = None, port: int | None = None) -> None:
    """Run the streamable-http MCP server under uvicorn (blocking)."""
    try:
        import uvicorn  # noqa: PLC0415 — optional extra
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    host = host or os.environ.get("PATENTKIT_MCP_HOST") or "127.0.0.1"
    if port is None:
        port = int(os.environ.get("PATENTKIT_MCP_PORT") or DEFAULT_PORT)
    token = os.environ.get("PATENTKIT_MCP_TOKEN") or None

    app = build_http_app(token=token)
    logger.info(
        "patentkit-mcp: streamable-http transport listening on http://%s:%d%s "
        "(bearer auth: %s)",
        host,
        port,
        MCP_PATH,
        "ON" if token else "OFF — set PATENTKIT_MCP_TOKEN before exposing this server publicly",
    )
    uvicorn.run(app, host=host, port=port, log_config=None)
