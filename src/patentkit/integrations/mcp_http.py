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
- ``PATENTKIT_AUTH_MODE``  ``token`` (default — the static bearer path above) or
  ``oauth-google`` (Google-delegated, email-allowlisted OAuth; see
  :mod:`patentkit.integrations.mcp_oauth` for its environment variables). In
  ``oauth-google`` mode the app mounts the OAuth metadata/authorize/token/
  register/revoke routes, a ``/auth/google/callback`` route, protected-resource
  metadata, and the SDK's bearer-auth middleware gating ``/mcp``.

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
GOOGLE_CALLBACK_PATH = "/auth/google/callback"

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


def _build_streamable_manager(
    toolset: Any, *, json_response: bool, stateless: bool
) -> tuple[Any, Any]:
    """Return ``(endpoint, lifespan)`` for the StreamableHTTP session manager."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: PLC0415

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

    return _StreamableHTTPEndpoint(), lifespan


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
        from starlette.applications import Starlette  # noqa: PLC0415
        from starlette.routing import Route  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    endpoint, lifespan = _build_streamable_manager(
        toolset, json_response=json_response, stateless=stateless
    )
    app: Any = Starlette(
        routes=[Route(MCP_PATH, endpoint=endpoint), Route("/", endpoint=endpoint)],
        lifespan=lifespan,
    )
    if token:
        app = BearerAuthMiddleware(app, token)
    return app


def build_oauth_http_app(
    toolset: Any = None,
    *,
    config: Any = None,
    provider: Any = None,
    json_response: bool = False,
    stateless: bool = False,
) -> Any:
    """Build the ASGI app for ``oauth-google`` mode (patentkit as OAuth AS).

    Mounts, alongside the StreamableHTTP ``/mcp`` route:

    - OAuth AS metadata + ``/authorize`` + ``/token`` + ``/register`` + ``/revoke``
      (:func:`mcp.server.auth.routes.create_auth_routes`),
    - ``/auth/google/callback`` (the upstream-IdP return leg),
    - RFC 9728 protected-resource metadata, and
    - the SDK bearer-auth middleware (``AuthenticationMiddleware`` +
      ``BearerAuthBackend``) plus ``RequireAuthMiddleware`` gating ``/mcp`` on a
      valid Google-derived access JWT.

    ``config``/``provider`` are injectable for tests; otherwise built from env.
    """
    try:
        from mcp.server.auth.middleware.auth_context import AuthContextMiddleware  # noqa: PLC0415
        from mcp.server.auth.middleware.bearer_auth import (  # noqa: PLC0415
            BearerAuthBackend,
            RequireAuthMiddleware,
        )
        from mcp.server.auth.routes import (  # noqa: PLC0415
            build_resource_metadata_url,
            create_auth_routes,
            create_protected_resource_routes,
        )
        from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions  # noqa: PLC0415
        from pydantic import AnyHttpUrl  # noqa: PLC0415
        from starlette.applications import Starlette  # noqa: PLC0415
        from starlette.middleware import Middleware  # noqa: PLC0415
        from starlette.middleware.authentication import AuthenticationMiddleware  # noqa: PLC0415
        from starlette.responses import HTMLResponse, RedirectResponse  # noqa: PLC0415
        from starlette.routing import Route  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    from patentkit.integrations.mcp_oauth import (  # noqa: PLC0415
        PatentkitGoogleOAuthProvider,
        PatentkitTokenVerifier,
        load_oauth_config,
    )

    if config is None:
        config = load_oauth_config()
    if provider is None:
        provider = PatentkitGoogleOAuthProvider(config)
    verifier = PatentkitTokenVerifier(provider)

    issuer_url = AnyHttpUrl(config.issuer)
    resource_url = AnyHttpUrl(config.resource_url)
    scopes = ["patentkit"]

    endpoint, lifespan = _build_streamable_manager(
        toolset, json_response=json_response, stateless=stateless
    )

    async def google_callback(request: Any) -> Any:
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        redirect_url, error = await provider.complete_google_callback(code, state)
        if redirect_url:
            return RedirectResponse(url=redirect_url, status_code=302, headers={"Cache-Control": "no-store"})
        if error == "access_denied":
            return HTMLResponse(
                "<h1>Not authorized</h1><p>This Google account is not on the "
                "allowlist for this patentkit server.</p>",
                status_code=403,
            )
        return HTMLResponse(
            "<h1>Sign-in failed</h1><p>The authorization request could not be "
            "completed. Please try connecting again.</p>",
            status_code=400,
        )

    resource_metadata_url = build_resource_metadata_url(resource_url)

    routes = create_auth_routes(
        provider=provider,
        issuer_url=issuer_url,
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=scopes, default_scopes=scopes
        ),
        revocation_options=RevocationOptions(enabled=True),
    )
    routes.append(Route(GOOGLE_CALLBACK_PATH, endpoint=google_callback, methods=["GET"]))
    routes.extend(
        create_protected_resource_routes(
            resource_url=resource_url,
            authorization_servers=[issuer_url],
            scopes_supported=scopes,
        )
    )
    gated = RequireAuthMiddleware(endpoint, scopes, resource_metadata_url)
    routes.append(Route(MCP_PATH, endpoint=gated))
    routes.append(Route("/", endpoint=gated))

    middleware = [
        Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(verifier)),
        Middleware(AuthContextMiddleware),
    ]

    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


def serve_http(*, host: str | None = None, port: int | None = None) -> None:
    """Run the streamable-http MCP server under uvicorn (blocking)."""
    try:
        import uvicorn  # noqa: PLC0415 — optional extra
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    host = host or os.environ.get("PATENTKIT_MCP_HOST") or "127.0.0.1"
    if port is None:
        port = int(os.environ.get("PATENTKIT_MCP_PORT") or DEFAULT_PORT)

    auth_mode = (os.environ.get("PATENTKIT_AUTH_MODE") or "token").strip().lower()
    if auth_mode == "oauth-google":
        app = build_oauth_http_app()
        logger.info(
            "patentkit-mcp: streamable-http transport listening on http://%s:%d%s "
            "(auth: oauth-google — Google sign-in, email allowlist)",
            host,
            port,
            MCP_PATH,
        )
    else:
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
