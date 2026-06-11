"""Google-delegated, email-allowlisted OAuth for the patentkit MCP HTTP server.

In ``oauth-google`` mode **patentkit is the OAuth 2.0 authorization server** that
claude.ai talks to (Dynamic Client Registration + authorization-code + PKCE), and
**Google is the upstream identity provider** that actually authenticates the human.
The flow::

    claude.ai  --DCR/authorize-->  patentkit (this module)  --redirect-->  Google
                                         ^                                    |
                                         |   /auth/google/callback            |
                                         +------------- id_token --------------+

On the Google callback we verify the id_token, check the user's email against an
allowlist, and only then mint *our own* tokens — short-lived access JWTs and
longer-lived refresh JWTs, both HS256-signed with ``PATENTKIT_OAUTH_SIGNING_SECRET``.
The SDK's bearer-auth middleware enforces :class:`PatentkitTokenVerifier` on ``/mcp``.

This implements the SDK ``OAuthAuthorizationServerProvider`` protocol
(``mcp.server.auth.provider``) plus the ``TokenVerifier`` protocol; see
:mod:`patentkit.integrations.mcp_http` for the Starlette wiring.

Required env (``oauth-google`` mode); fails fast if any is missing:

- ``PATENTKIT_AUTH_MODE``          ``token`` (default) | ``oauth-google``
- ``PATENTKIT_PUBLIC_URL``         issuer/resource base, e.g.
  ``https://patentkit-mcp-xxx-uc.a.run.app`` (HTTPS; no trailing slash needed)
- ``GOOGLE_OAUTH_CLIENT_ID``       our Google OAuth client id
- ``GOOGLE_OAUTH_CLIENT_SECRET``   our Google OAuth client secret
- ``PATENTKIT_ALLOWED_EMAILS``     comma-separated allowlist (lowercased, exact)
- ``PATENTKIT_OAUTH_SIGNING_SECRET``  HS256 secret for tokens patentkit issues
- ``PATENTKIT_OAUTH_DIR``          (optional) dir for persisted clients/denylist;
  defaults to ``<PATENTKIT_SESSION_DIR>/../oauth`` or ``/data/oauth``.

The Google redirect URI that must be registered in the Google Cloud console is
``{PATENTKIT_PUBLIC_URL}/auth/google/callback``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

GOOGLE_CALLBACK_PATH = "/auth/google/callback"

ACCESS_TOKEN_TTL = 3600  # 1 hour
REFRESH_TOKEN_TTL = 30 * 24 * 3600  # 30 days
AUTH_CODE_TTL = 300  # 5 minutes
PENDING_TTL = 600  # 10 minutes — round-trip to Google must finish in this window

DEFAULT_SCOPE = "patentkit"


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class OAuthConfig:
    """Validated configuration for ``oauth-google`` mode."""

    public_url: str  # normalized, no trailing slash
    google_client_id: str
    google_client_secret: str
    allowed_emails: frozenset[str]
    signing_secret: str
    oauth_dir: Path

    @property
    def issuer(self) -> str:
        return self.public_url

    @property
    def google_redirect_uri(self) -> str:
        return f"{self.public_url}{GOOGLE_CALLBACK_PATH}"

    @property
    def resource_url(self) -> str:
        return f"{self.public_url}/mcp"

    @property
    def clients_path(self) -> Path:
        return self.oauth_dir / "clients.json"

    @property
    def denylist_path(self) -> Path:
        return self.oauth_dir / "denylist.json"


def _default_oauth_dir(env: dict[str, str]) -> Path:
    session_dir = env.get("PATENTKIT_SESSION_DIR")
    if session_dir:
        # sibling of the session dir, e.g. /data/sessions -> /data/oauth
        return Path(session_dir).resolve().parent / "oauth"
    return Path("/data/oauth")


def load_oauth_config(env: dict[str, str] | None = None) -> OAuthConfig:
    """Build and validate :class:`OAuthConfig` from environment variables.

    Raises ``ValueError`` listing every missing required value (fail fast).
    """
    env = dict(os.environ if env is None else env)

    missing: list[str] = []

    def _req(key: str) -> str:
        value = (env.get(key) or "").strip()
        if not value:
            missing.append(key)
        return value

    public_url = _req("PATENTKIT_PUBLIC_URL").rstrip("/")
    google_client_id = _req("GOOGLE_OAUTH_CLIENT_ID")
    google_client_secret = _req("GOOGLE_OAUTH_CLIENT_SECRET")
    signing_secret = _req("PATENTKIT_OAUTH_SIGNING_SECRET")

    raw_emails = (env.get("PATENTKIT_ALLOWED_EMAILS") or "").strip()
    allowed = frozenset(
        e.strip().lower() for e in raw_emails.split(",") if e.strip()
    )
    if not allowed:
        missing.append("PATENTKIT_ALLOWED_EMAILS")

    if missing:
        raise ValueError(
            "PATENTKIT_AUTH_MODE=oauth-google requires these environment "
            f"variables, which are unset or empty: {', '.join(sorted(set(missing)))}"
        )

    if not public_url.startswith("https://") and "localhost" not in public_url and "127.0.0.1" not in public_url:
        raise ValueError(
            f"PATENTKIT_PUBLIC_URL must be HTTPS (got {public_url!r})"
        )

    oauth_dir_env = (env.get("PATENTKIT_OAUTH_DIR") or "").strip()
    oauth_dir = Path(oauth_dir_env).resolve() if oauth_dir_env else _default_oauth_dir(env)

    return OAuthConfig(
        public_url=public_url,
        google_client_id=google_client_id,
        google_client_secret=google_client_secret,
        allowed_emails=allowed,
        signing_secret=signing_secret,
        oauth_dir=oauth_dir,
    )


# --------------------------------------------------------------------------- helpers


def _now() -> int:
    return int(time.time())


@dataclass
class _Pending:
    """A claude.ai authorization request parked while we round-trip to Google."""

    client_id: str
    redirect_uri: str
    code_challenge: str
    scopes: list[str]
    client_state: str | None
    resource: str | None
    created_at: int = field(default_factory=_now)


@dataclass
class _OurAuthCode:
    """Our authorization code, bound to the claude.ai client + its PKCE challenge."""

    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    scopes: list[str]
    subject: str  # the verified, allowlisted email
    resource: str | None
    expires_at: int


# --------------------------------------------------------------------------- provider


class PatentkitGoogleOAuthProvider:
    """``OAuthAuthorizationServerProvider`` where Google is the IdP.

    Generic over the SDK's ``AuthorizationCode``/``RefreshToken``/``AccessToken``
    types. Client registrations and the revocation denylist persist to the oauth
    dir (Cloud Run scale-to-zero restarts otherwise drop registrations). Pending
    authorizations are kept **in memory only**: the browser round-trip to Google
    completes within one warm-instance lifetime, and persisting half-finished
    login state buys little while adding disk churn. The denylist is small and
    keyed by ``jti``.
    """

    def __init__(self, config: OAuthConfig, *, http_client: Any = None) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}
        self._pending: dict[str, _Pending] = {}
        self._codes: dict[str, _OurAuthCode] = {}
        self._denylist: set[str] = set()
        # Injectable for tests; created lazily otherwise so import stays cheap.
        self._http_client = http_client
        config.oauth_dir.mkdir(parents=True, exist_ok=True)
        self._load_clients()
        self._load_denylist()

    # -- SDK type accessors (imported lazily; mcp is an optional extra) -------

    @staticmethod
    def _provider_types() -> Any:
        from mcp.server.auth.provider import (  # noqa: PLC0415
            AccessToken,
            AuthorizationCode,
            RefreshToken,
        )

        return AccessToken, AuthorizationCode, RefreshToken

    # -- persistence ---------------------------------------------------------

    def _load_clients(self) -> None:
        from mcp.shared.auth import OAuthClientInformationFull  # noqa: PLC0415

        path = self.config.clients_path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - corrupt file
            logger.warning("oauth: could not read %s: %s", path, exc)
            return
        for client_id, blob in (raw or {}).items():
            try:
                self._clients[client_id] = OAuthClientInformationFull.model_validate(blob)
            except Exception as exc:  # pragma: no cover - skip bad rows
                logger.warning("oauth: skipping invalid client %s: %s", client_id, exc)

    def _persist_clients(self) -> None:
        path = self.config.clients_path
        blob = {cid: json.loads(c.model_dump_json()) for cid, c in self._clients.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=2))
        tmp.replace(path)

    def _load_denylist(self) -> None:
        path = self.config.denylist_path
        if not path.exists():
            return
        try:
            self._denylist = set(json.loads(path.read_text()) or [])
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            logger.warning("oauth: could not read denylist %s: %s", path, exc)

    def _persist_denylist(self) -> None:
        path = self.config.denylist_path
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self._denylist), indent=2))
        tmp.replace(path)

    # -- DCR -----------------------------------------------------------------

    async def get_client(self, client_id: str) -> Any | None:
        with self._lock:
            return self._clients.get(client_id)

    async def register_client(self, client_info: Any) -> None:
        with self._lock:
            self._clients[str(client_info.client_id)] = client_info
            self._persist_clients()

    # -- authorize (redirect the user to Google) -----------------------------

    async def authorize(self, client: Any, params: Any) -> str:
        state = secrets.token_urlsafe(32)
        pending = _Pending(
            client_id=str(client.client_id),
            redirect_uri=str(params.redirect_uri),
            code_challenge=params.code_challenge,
            scopes=list(params.scopes or [DEFAULT_SCOPE]),
            client_state=params.state,
            resource=params.resource,
        )
        with self._lock:
            self._gc_pending()
            self._pending[state] = pending

        query = urlencode(
            {
                "client_id": self.config.google_client_id,
                "redirect_uri": self.config.google_redirect_uri,
                "response_type": "code",
                "scope": "openid email",
                "state": state,
                "access_type": "online",
                "prompt": "select_account",
            }
        )
        return f"{GOOGLE_AUTH_ENDPOINT}?{query}"

    def _gc_pending(self) -> None:
        cutoff = _now() - PENDING_TTL
        for key in [k for k, v in self._pending.items() if v.created_at < cutoff]:
            self._pending.pop(key, None)

    # -- Google callback (called by the Starlette route) ---------------------

    async def complete_google_callback(self, code: str | None, state: str | None) -> tuple[str, str | None]:
        """Finish the upstream exchange and return ``(redirect_url, error)``.

        On success ``error`` is ``None`` and ``redirect_url`` points back at the
        claude.ai redirect_uri carrying our freshly-minted authorization code.
        On a non-allowlisted/unverified email, ``error`` is ``"access_denied"``
        and the redirect carries the OAuth error back to the client (if we can
        recover its redirect_uri); otherwise the route renders an HTML notice.
        """
        if not state or not code:
            return ("", "invalid_request")

        with self._lock:
            pending = self._pending.pop(state, None)
        if pending is None:
            return ("", "invalid_request")

        try:
            email, verified = await self._verify_google_code(code)
        except Exception as exc:  # pragma: no cover - network/verification failure
            logger.warning("oauth: google verification failed: %s", exc)
            return (self._error_redirect(pending, "server_error"), "server_error")

        if not verified or email is None or email.lower() not in self.config.allowed_emails:
            logger.info("oauth: denying login for email=%r verified=%s", email, verified)
            return (self._error_redirect(pending, "access_denied"), "access_denied")

        our_code = _OurAuthCode(
            code=secrets.token_urlsafe(32),
            client_id=pending.client_id,
            redirect_uri=pending.redirect_uri,
            code_challenge=pending.code_challenge,
            scopes=pending.scopes,
            subject=email.lower(),
            resource=pending.resource,
            expires_at=_now() + AUTH_CODE_TTL,
        )
        with self._lock:
            self._codes[our_code.code] = our_code

        params = {"code": our_code.code}
        if pending.client_state is not None:
            params["state"] = pending.client_state
        sep = "&" if "?" in pending.redirect_uri else "?"
        return (f"{pending.redirect_uri}{sep}{urlencode(params)}", None)

    def _error_redirect(self, pending: _Pending, error: str) -> str:
        if not pending.redirect_uri:
            return ""
        params = {"error": error}
        if pending.client_state is not None:
            params["state"] = pending.client_state
        sep = "&" if "?" in pending.redirect_uri else "?"
        return f"{pending.redirect_uri}{sep}{urlencode(params)}"

    async def _verify_google_code(self, code: str) -> tuple[str | None, bool]:
        """Exchange the Google code and verify the returned id_token.

        Returns ``(email, email_verified)``. Tests monkeypatch this method (or the
        two helpers it calls) to avoid network.
        """
        id_token = await self._exchange_google_code(code)
        claims = self._verify_google_id_token(id_token)
        email = claims.get("email")
        verified = bool(claims.get("email_verified"))
        return (email, verified)

    async def _exchange_google_code(self, code: str) -> str:
        data = {
            "code": code,
            "client_id": self.config.google_client_id,
            "client_secret": self.config.google_client_secret,
            "redirect_uri": self.config.google_redirect_uri,
            "grant_type": "authorization_code",
        }
        client = self._http_client
        if client is not None:
            resp = await client.post(GOOGLE_TOKEN_ENDPOINT, data=data)
            resp.raise_for_status()
            payload = resp.json()
        else:  # pragma: no cover - real network path
            async with httpx.AsyncClient(timeout=15) as ac:
                resp = await ac.post(GOOGLE_TOKEN_ENDPOINT, data=data)
                resp.raise_for_status()
                payload = resp.json()
        id_token = payload.get("id_token")
        if not id_token:
            raise ValueError("google token response missing id_token")
        return id_token

    def _verify_google_id_token(self, id_token: str) -> dict[str, Any]:
        import jwt  # noqa: PLC0415
        from jwt import PyJWKClient  # noqa: PLC0415

        jwks = PyJWKClient(GOOGLE_JWKS_URI)
        signing_key = jwks.get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self.config.google_client_id,
            issuer=list(GOOGLE_ISSUERS),
            options={"require": ["exp", "iss", "aud"]},
        )
        return claims

    # -- token issuance ------------------------------------------------------

    def _mint_access_token(self, subject: str, scopes: list[str], client_id: str) -> tuple[str, int]:
        import jwt  # noqa: PLC0415

        now = _now()
        exp = now + ACCESS_TOKEN_TTL
        payload = {
            "sub": subject,
            "iss": self.config.issuer,
            "aud": self.config.resource_url,
            "iat": now,
            "exp": exp,
            "scope": " ".join(scopes),
            "client_id": client_id,
            "jti": secrets.token_urlsafe(16),
            "type": "access",
        }
        token = jwt.encode(payload, self.config.signing_secret, algorithm="HS256")
        return token, exp

    def _mint_refresh_token(self, subject: str, scopes: list[str], client_id: str) -> str:
        import jwt  # noqa: PLC0415

        now = _now()
        payload = {
            "sub": subject,
            "iss": self.config.issuer,
            "aud": self.config.resource_url,
            "iat": now,
            "exp": now + REFRESH_TOKEN_TTL,
            "scope": " ".join(scopes),
            "client_id": client_id,
            "jti": secrets.token_urlsafe(16),
            "type": "refresh",
        }
        return jwt.encode(payload, self.config.signing_secret, algorithm="HS256")

    def _decode_our_jwt(self, token: str, *, expected_type: str) -> dict[str, Any] | None:
        import jwt  # noqa: PLC0415

        try:
            claims = jwt.decode(
                token,
                self.config.signing_secret,
                algorithms=["HS256"],
                audience=self.config.resource_url,
                issuer=self.config.issuer,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.InvalidTokenError:
            return None
        if claims.get("type") != expected_type:
            return None
        if claims.get("jti") in self._denylist:
            return None
        return claims

    def _oauth_token(self, subject: str, scopes: list[str], client_id: str) -> Any:
        from mcp.shared.auth import OAuthToken  # noqa: PLC0415

        access, exp = self._mint_access_token(subject, scopes, client_id)
        refresh = self._mint_refresh_token(subject, scopes, client_id)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=exp - _now(),
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    # -- authorization code grant -------------------------------------------

    async def load_authorization_code(self, client: Any, authorization_code: str) -> Any | None:
        _, AuthorizationCode, _ = self._provider_types()
        from pydantic import AnyUrl  # noqa: PLC0415

        with self._lock:
            record = self._codes.get(authorization_code)
        if record is None or record.client_id != str(client.client_id):
            return None
        if record.expires_at < _now():
            return None
        return AuthorizationCode(
            code=record.code,
            scopes=record.scopes,
            expires_at=float(record.expires_at),
            client_id=record.client_id,
            code_challenge=record.code_challenge,
            redirect_uri=AnyUrl(record.redirect_uri),
            redirect_uri_provided_explicitly=True,
            resource=record.resource,
            subject=record.subject,
        )

    async def exchange_authorization_code(self, client: Any, authorization_code: Any) -> Any:
        # The SDK's TokenHandler already verified PKCE (code_verifier vs
        # code_challenge, S256) and expiry before calling us; consume the code.
        from mcp.server.auth.provider import TokenError  # noqa: PLC0415

        with self._lock:
            record = self._codes.pop(authorization_code.code, None)
        if record is None:
            raise TokenError("invalid_grant", "authorization code does not exist")
        subject = authorization_code.subject or record.subject
        if subject.lower() not in self.config.allowed_emails:
            raise TokenError("invalid_grant", "email no longer authorized")
        return self._oauth_token(subject.lower(), authorization_code.scopes, str(client.client_id))

    # -- refresh-token grant -------------------------------------------------

    async def load_refresh_token(self, client: Any, refresh_token: str) -> Any | None:
        _, _, RefreshToken = self._provider_types()
        claims = self._decode_our_jwt(refresh_token, expected_type="refresh")
        if claims is None or claims.get("client_id") != str(client.client_id):
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=str(client.client_id),
            scopes=(claims.get("scope") or "").split() or [DEFAULT_SCOPE],
            expires_at=int(claims["exp"]),
            subject=claims.get("sub"),
        )

    async def exchange_refresh_token(self, client: Any, refresh_token: Any, scopes: list[str]) -> Any:
        from mcp.server.auth.provider import TokenError  # noqa: PLC0415

        subject = (refresh_token.subject or "").lower()
        if not subject or subject not in self.config.allowed_emails:
            raise TokenError("invalid_grant", "email no longer authorized")
        new_scopes = scopes or refresh_token.scopes
        # Rotate: best-effort revoke the presented refresh token's jti.
        claims = self._decode_our_jwt(refresh_token.token, expected_type="refresh")
        if claims and claims.get("jti"):
            with self._lock:
                self._denylist.add(claims["jti"])
                self._persist_denylist()
        return self._oauth_token(subject, list(new_scopes), str(client.client_id))

    # -- introspection / revocation -----------------------------------------

    async def load_access_token(self, token: str) -> Any | None:
        AccessToken, _, _ = self._provider_types()
        claims = self._decode_our_jwt(token, expected_type="access")
        if claims is None:
            return None
        subject = (claims.get("sub") or "").lower()
        if subject not in self.config.allowed_emails:
            return None
        return AccessToken(
            token=token,
            client_id=str(claims.get("client_id") or ""),
            scopes=(claims.get("scope") or "").split() or [DEFAULT_SCOPE],
            expires_at=int(claims["exp"]),
            resource=self.config.resource_url,
            subject=subject,
            claims={"iss": claims.get("iss")},
        )

    async def revoke_token(self, token: Any) -> None:
        # Stateless JWTs can't be truly revoked; we keep a small persisted
        # jti-denylist (checked in _decode_our_jwt) for best-effort revocation.
        # Limitation: an access token whose jti we never learn (only its string
        # was presented) is still denylisted here because we decode it to read
        # the jti; tokens already expired are simply ignored.
        raw = getattr(token, "token", None)
        if not raw:
            return
        import jwt  # noqa: PLC0415

        try:
            claims = jwt.decode(raw, self.config.signing_secret, algorithms=["HS256"], options={"verify_aud": False})
        except jwt.InvalidTokenError:
            return
        jti = claims.get("jti")
        if jti:
            with self._lock:
                self._denylist.add(jti)
                self._persist_denylist()


# --------------------------------------------------------------------------- verifier


class PatentkitTokenVerifier:
    """``TokenVerifier`` backed by the provider's access-JWT validation."""

    def __init__(self, provider: PatentkitGoogleOAuthProvider) -> None:
        self.provider = provider

    async def verify_token(self, token: str) -> Any | None:
        return await self.provider.load_access_token(token)
