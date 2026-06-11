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
- ``PATENTKIT_AUTH_MODE``  one of three modes (code-level default ``token``):

  - ``token``         the static bearer path above (``PATENTKIT_MCP_TOKEN``).
  - ``oauth-google``  Google-delegated, email-allowlisted OAuth — patentkit is
    the OAuth AS, Google authenticates the human. Mounts the OAuth metadata/
    authorize/token/register/revoke routes, a ``/auth/google/callback`` route,
    protected-resource metadata, and the SDK bearer-auth middleware gating
    ``/mcp``.
  - ``oauth-secret``  self-contained OAuth with NO external identity provider —
    patentkit is the OAuth AS and the human authenticates by typing a shared
    secret (``PATENTKIT_ACCESS_SECRET``) once on a local ``/auth/approve`` page.
    Same OAuth surface as ``oauth-google`` but with the approve GET+POST routes
    mounted instead of the Google callback, and zero outbound network calls.

  See :mod:`patentkit.integrations.mcp_oauth` for each mode's environment
  variables.

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
APPROVE_PATH = "/auth/approve"

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


def _google_auth_routes(provider: Any) -> list[Any]:
    """The Google-mode user-authentication leg: a single GET callback route."""
    from starlette.responses import HTMLResponse, RedirectResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

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

    return [Route(GOOGLE_CALLBACK_PATH, endpoint=google_callback, methods=["GET"])]


def _approve_page_html(state: str, *, error: str | None = None) -> str:
    """Minimal self-contained approve page — no external assets/CDN, inline CSS."""
    from html import escape  # noqa: PLC0415

    state_attr = escape(state, quote=True)
    error_html = (
        f'<p class="err" role="alert">{escape(error)}</p>' if error else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorize patentkit</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          background: #f5f5f7; margin: 0; padding: 2rem;
          display: flex; justify-content: center; }}
  .card {{ background: #fff; max-width: 26rem; width: 100%; padding: 2rem;
           border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .5rem; }}
  p {{ color: #444; line-height: 1.5; }}
  label {{ display: block; font-weight: 600; margin: 1rem 0 .35rem; }}
  input[type=password] {{ width: 100%; box-sizing: border-box; padding: .6rem;
           font-size: 1rem; border: 1px solid #ccc; border-radius: 8px; }}
  button {{ margin-top: 1.25rem; width: 100%; padding: .7rem; font-size: 1rem;
            font-weight: 600; color: #fff; background: #1a73e8; border: 0;
            border-radius: 8px; cursor: pointer; }}
  .err {{ color: #b00020; font-weight: 600; }}
</style>
</head>
<body>
  <main class="card">
    <h1>Authorize this patentkit MCP server</h1>
    <p>Enter the access code to connect this patentkit server to claude.ai.</p>
    {error_html}
    <form method="post" action="{APPROVE_PATH}">
      <input type="hidden" name="state" value="{state_attr}">
      <label for="secret">Access code</label>
      <input type="password" id="secret" name="secret" autocomplete="off"
             autofocus required>
      <button type="submit">Authorize</button>
    </form>
  </main>
</body>
</html>"""


def _secret_auth_routes(provider: Any) -> list[Any]:
    """The secret-mode user-authentication leg: GET + POST ``/auth/approve``."""
    from starlette.responses import HTMLResponse, RedirectResponse  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    async def approve_get(request: Any) -> Any:
        state = request.query_params.get("state")
        if not provider.peek_pending(state):
            return HTMLResponse(
                "<h1>Request expired</h1><p>This authorization request is "
                "unknown or has expired. Please try connecting again.</p>",
                status_code=400,
            )
        return HTMLResponse(_approve_page_html(state), headers={"Cache-Control": "no-store"})

    async def approve_post(request: Any) -> Any:
        form = await request.form()
        state = form.get("state")
        secret = form.get("secret")
        redirect_url, error = provider.complete_secret_approval(state, secret)
        if redirect_url:
            return RedirectResponse(url=redirect_url, status_code=303, headers={"Cache-Control": "no-store"})
        if error == "access_denied":
            # Wrong secret: re-render the form with an error, issue nothing.
            return HTMLResponse(
                _approve_page_html(state or "", error="Incorrect access code."),
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        # Unknown/expired state.
        return HTMLResponse(
            "<h1>Request expired</h1><p>This authorization request is unknown "
            "or has expired. Please try connecting again.</p>",
            status_code=400,
        )

    return [
        Route(APPROVE_PATH, endpoint=approve_get, methods=["GET"]),
        Route(APPROVE_PATH, endpoint=approve_post, methods=["POST"]),
    ]


def _assemble_oauth_app(
    *,
    config: Any,
    provider: Any,
    auth_route_builder: Any,
    toolset: Any,
    json_response: bool,
    stateless: bool,
) -> Any:
    """Build the OAuth AS Starlette app shared by both OAuth modes.

    Everything is identical across modes except ``auth_route_builder(provider)``,
    which supplies the mode-specific user-authentication route(s) (Google
    callback vs. the local approve GET+POST).
    """
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
    from starlette.routing import Route  # noqa: PLC0415

    from patentkit.integrations.mcp_oauth import PatentkitTokenVerifier  # noqa: PLC0415

    verifier = PatentkitTokenVerifier(provider)
    issuer_url = AnyHttpUrl(config.issuer)
    resource_url = AnyHttpUrl(config.resource_url)
    scopes = ["patentkit"]

    endpoint, lifespan = _build_streamable_manager(
        toolset, json_response=json_response, stateless=stateless
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
    routes.extend(auth_route_builder(provider))
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


def build_oauth_http_app(
    toolset: Any = None,
    *,
    config: Any = None,
    provider: Any = None,
    json_response: bool = False,
    stateless: bool = False,
) -> Any:
    """Build the ASGI app for an OAuth mode (patentkit as the OAuth AS).

    Mounts, alongside the StreamableHTTP ``/mcp`` route:

    - OAuth AS metadata + ``/authorize`` + ``/token`` + ``/register`` + ``/revoke``
      (:func:`mcp.server.auth.routes.create_auth_routes`),
    - the mode's user-authentication leg — ``/auth/google/callback`` for
      ``oauth-google`` or the ``/auth/approve`` GET+POST page for ``oauth-secret``,
    - RFC 9728 protected-resource metadata, and
    - the SDK bearer-auth middleware (``AuthenticationMiddleware`` +
      ``BearerAuthBackend``) plus ``RequireAuthMiddleware`` gating ``/mcp`` on a
      valid patentkit-issued access JWT.

    The mode is taken from ``config.mode`` (built from env when ``config`` is
    ``None``). ``config``/``provider`` are injectable for tests.
    """
    try:
        import mcp.server.auth.routes  # noqa: F401, PLC0415
        import starlette.applications  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    from patentkit.integrations.mcp_oauth import (  # noqa: PLC0415
        PatentkitGoogleOAuthProvider,
        PatentkitSecretOAuthProvider,
        load_oauth_config,
    )

    if config is None:
        config = load_oauth_config()

    if config.mode == "oauth-secret":
        if provider is None:
            provider = PatentkitSecretOAuthProvider(config)
        auth_route_builder = _secret_auth_routes
    else:
        if provider is None:
            provider = PatentkitGoogleOAuthProvider(config)
        auth_route_builder = _google_auth_routes

    return _assemble_oauth_app(
        config=config,
        provider=provider,
        auth_route_builder=auth_route_builder,
        toolset=toolset,
        json_response=json_response,
        stateless=stateless,
    )


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
    elif auth_mode == "oauth-secret":
        app = build_oauth_http_app()
        logger.info(
            "patentkit-mcp: streamable-http transport listening on http://%s:%d%s "
            "(auth: oauth-secret — self-contained, shared access code, no external IdP)",
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
