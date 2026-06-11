"""Offline tests for Google-delegated, email-allowlisted MCP OAuth (no network).

Google's token exchange and id_token verification are monkeypatched, so nothing
here touches the network. Http-app tests are guarded with ``importorskip`` so
the suite still runs with core deps only.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time

import pytest

from patentkit.integrations.mcp_oauth import (
    SECRET_SUBJECT,
    PatentkitGoogleOAuthProvider,
    PatentkitSecretOAuthProvider,
    PatentkitTokenVerifier,
    load_oauth_config,
)

mcp_auth = pytest.importorskip("mcp.server.auth.provider", reason="requires the mcp-http extra")
shared_auth = pytest.importorskip("mcp.shared.auth", reason="requires the mcp-http extra")
AuthorizationParams = mcp_auth.AuthorizationParams
TokenError = mcp_auth.TokenError
OAuthClientInformationFull = shared_auth.OAuthClientInformationFull


# --------------------------------------------------------------------------- fixtures


def _env(tmp_path, emails="alice@example.com, bob@example.com"):
    return {
        "PATENTKIT_PUBLIC_URL": "https://patentkit.example.run.app",
        "GOOGLE_OAUTH_CLIENT_ID": "google-client-id",
        "GOOGLE_OAUTH_CLIENT_SECRET": "google-secret",
        "PATENTKIT_ALLOWED_EMAILS": emails,
        "PATENTKIT_OAUTH_SIGNING_SECRET": "super-secret-signing-key-at-least-32b!",
        "PATENTKIT_OAUTH_DIR": str(tmp_path / "oauth"),
    }


def _make_provider(tmp_path, emails="alice@example.com, bob@example.com"):
    config = load_oauth_config(_env(tmp_path, emails))
    return PatentkitGoogleOAuthProvider(config)


def _client():
    return OAuthClientInformationFull(
        client_id="claude-client-1",
        client_secret="cs",
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="patentkit",
    )


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def _auth_params(challenge, *, state="client-state-xyz"):
    from pydantic import AnyUrl

    return AuthorizationParams(
        state=state,
        scopes=["patentkit"],
        code_challenge=challenge,
        redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
        redirect_uri_provided_explicitly=True,
        resource="https://patentkit.example.run.app/mcp",
    )


def _patch_google(provider, monkeypatch, email, *, verified=True):
    async def fake_verify(code):
        return (email, verified)

    monkeypatch.setattr(provider, "_verify_google_code", fake_verify)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- config


def test_config_fails_fast_on_missing(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_oauth_config({"PATENTKIT_PUBLIC_URL": "https://x.run.app"})
    msg = str(exc.value)
    assert "GOOGLE_OAUTH_CLIENT_ID" in msg
    assert "PATENTKIT_ALLOWED_EMAILS" in msg


def test_config_normalizes(tmp_path):
    config = load_oauth_config(_env(tmp_path))
    assert config.issuer == "https://patentkit.example.run.app"
    assert config.google_redirect_uri == "https://patentkit.example.run.app/auth/google/callback"
    assert config.resource_url == "https://patentkit.example.run.app/mcp"
    assert config.allowed_emails == frozenset({"alice@example.com", "bob@example.com"})


# --------------------------------------------------------------------------- DCR


def test_register_and_load_roundtrip_and_persist(tmp_path):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))

    loaded = run(provider.get_client("claude-client-1"))
    assert loaded is not None and loaded.client_id == "claude-client-1"

    # A fresh provider over the same dir reloads the persisted registration.
    provider2 = _make_provider(tmp_path)
    reloaded = run(provider2.get_client("claude-client-1"))
    assert reloaded is not None
    assert str(reloaded.redirect_uris[0]) == "https://claude.ai/api/mcp/auth_callback"


# --------------------------------------------------------------------------- authorize


def test_authorize_redirects_to_google_with_state(tmp_path):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()

    url = run(provider.authorize(client, _auth_params(challenge)))
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(url).query)
    assert q["client_id"] == ["google-client-id"]
    assert q["redirect_uri"] == ["https://patentkit.example.run.app/auth/google/callback"]
    assert q["scope"] == ["openid email"]
    assert q["response_type"] == ["code"]
    state = q["state"][0]
    assert state in provider._pending


# --------------------------------------------------------------------------- callback


def test_callback_allowlisted_issues_code_and_redirects(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    verifier, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]

    _patch_google(provider, monkeypatch, "alice@example.com")
    redirect, error = run(provider.complete_google_callback("g-code", state))
    assert error is None
    parsed = parse_qs(urlparse(redirect).query)
    assert redirect.startswith("https://claude.ai/api/mcp/auth_callback?")
    assert parsed["state"] == ["client-state-xyz"]
    our_code = parsed["code"][0]
    assert our_code in provider._codes


def test_callback_non_allowlisted_denies(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]

    _patch_google(provider, monkeypatch, "evil@elsewhere.com")
    redirect, error = run(provider.complete_google_callback("g-code", state))
    assert error == "access_denied"
    assert "error=access_denied" in redirect
    assert provider._codes == {}  # nothing issued


def test_callback_unverified_email_denies(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]

    _patch_google(provider, monkeypatch, "alice@example.com", verified=False)
    _, error = run(provider.complete_google_callback("g-code", state))
    assert error == "access_denied"
    assert provider._codes == {}


def test_callback_unknown_state_rejected(tmp_path):
    provider = _make_provider(tmp_path)
    _, error = run(provider.complete_google_callback("g-code", "no-such-state"))
    assert error == "invalid_request"


# --------------------------------------------------------------------------- token grant


def _full_to_code(tmp_path, monkeypatch, email="alice@example.com"):
    provider = _make_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    verifier, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]
    _patch_google(provider, monkeypatch, email)
    redirect, _ = run(provider.complete_google_callback("g-code", state))
    our_code = parse_qs(urlparse(redirect).query)["code"][0]
    return provider, client, our_code, verifier


def test_exchange_authorization_code_issues_jwt(tmp_path, monkeypatch):
    provider, client, code, verifier = _full_to_code(tmp_path, monkeypatch)
    # The SDK loads then exchanges; mirror that.
    auth_code = run(provider.load_authorization_code(client, code))
    assert auth_code is not None
    assert auth_code.subject == "alice@example.com"
    token = run(provider.exchange_authorization_code(client, auth_code))
    assert token.access_token and token.refresh_token
    assert token.token_type == "Bearer"

    access = run(provider.load_access_token(token.access_token))
    assert access is not None and access.subject == "alice@example.com"
    assert "patentkit" in access.scopes


def test_pkce_mismatch_rejected_by_sdk_handler(tmp_path, monkeypatch):
    # The SDK's TokenHandler does the PKCE check before exchange. We assert the
    # stored code_challenge matches the issued auth code so the SDK check is
    # meaningful, and that a WRONG verifier would not hash to it.
    provider, client, code, verifier = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    good = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert auth_code.code_challenge == good
    wrong = base64.urlsafe_b64encode(hashlib.sha256(b"not-the-verifier").digest()).decode().rstrip("=")
    assert wrong != auth_code.code_challenge


def test_authorization_code_single_use(tmp_path, monkeypatch):
    provider, client, code, verifier = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    run(provider.exchange_authorization_code(client, auth_code))
    # Code is consumed; a second exchange fails.
    with pytest.raises(TokenError):
        run(provider.exchange_authorization_code(client, auth_code))


# --------------------------------------------------------------------------- verify_token


def test_verify_token_accepts_fresh_rejects_tampered(tmp_path, monkeypatch):
    provider, client, code, _ = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    tv = PatentkitTokenVerifier(provider)

    assert run(tv.verify_token(token.access_token)) is not None
    # Tamper with the signature.
    tampered = token.access_token[:-3] + ("aaa" if token.access_token[-3:] != "aaa" else "bbb")
    assert run(tv.verify_token(tampered)) is None
    assert run(tv.verify_token("not-a-jwt")) is None


def test_verify_token_rejects_expired(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path)
    # Mint an already-expired access token directly.
    import jwt

    now = int(time.time())
    payload = {
        "sub": "alice@example.com",
        "iss": provider.config.issuer,
        "aud": provider.config.resource_url,
        "iat": now - 100,
        "exp": now - 10,
        "scope": "patentkit",
        "client_id": "c",
        "jti": "x",
        "type": "access",
    }
    expired = jwt.encode(payload, provider.config.signing_secret, algorithm="HS256")
    assert run(provider.load_access_token(expired)) is None


def test_verify_token_rejects_deallowlisted_email(tmp_path, monkeypatch):
    provider, client, code, _ = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    # Simulate the email being removed from the allowlist.
    object.__setattr__(provider.config, "allowed_emails", frozenset({"someoneelse@example.com"}))
    assert run(provider.load_access_token(token.access_token)) is None


# --------------------------------------------------------------------------- refresh


def test_exchange_refresh_token_rotates(tmp_path, monkeypatch):
    provider, client, code, _ = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))

    rt = run(provider.load_refresh_token(client, token.refresh_token))
    assert rt is not None and rt.subject == "alice@example.com"
    new = run(provider.exchange_refresh_token(client, rt, rt.scopes))
    assert new.access_token != token.access_token
    assert run(provider.load_access_token(new.access_token)) is not None
    # Old refresh token is denylisted after rotation.
    assert run(provider.load_refresh_token(client, token.refresh_token)) is None


def test_refresh_rechecks_allowlist(tmp_path, monkeypatch):
    provider, client, code, _ = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    rt = run(provider.load_refresh_token(client, token.refresh_token))
    object.__setattr__(provider.config, "allowed_emails", frozenset())
    with pytest.raises(TokenError):
        run(provider.exchange_refresh_token(client, rt, rt.scopes))


# --------------------------------------------------------------------------- revocation


def test_revoke_denylists_access_token(tmp_path, monkeypatch):
    provider, client, code, _ = _full_to_code(tmp_path, monkeypatch)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    access = run(provider.load_access_token(token.access_token))
    assert access is not None
    run(provider.revoke_token(access))
    assert run(provider.load_access_token(token.access_token)) is None


# --------------------------------------------------------------------------- app

starlette_testclient = pytest.importorskip("starlette.testclient", reason="requires the mcp-http extra")
TestClient = starlette_testclient.TestClient

from patentkit.integrations.mcp_http import build_oauth_http_app  # noqa: E402
from patentkit.integrations.toolset import PatentToolset  # noqa: E402

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


def _oauth_app(tmp_path):
    config = load_oauth_config(_env(tmp_path))
    provider = PatentkitGoogleOAuthProvider(config)
    app = build_oauth_http_app(PatentToolset(), config=config, provider=provider, json_response=True)
    return app, provider


def test_metadata_advertises_endpoints(tmp_path):
    app, _ = _oauth_app(tmp_path)
    with TestClient(app) as client:
        meta = client.get("/.well-known/oauth-authorization-server").json()
        assert meta["issuer"] == "https://patentkit.example.run.app/"
        assert meta["authorization_endpoint"].endswith("/authorize")
        assert meta["token_endpoint"].endswith("/token")
        assert meta["registration_endpoint"].endswith("/register")
        assert meta["revocation_endpoint"].endswith("/revoke")


def test_unauthenticated_mcp_is_401(tmp_path):
    app, _ = _oauth_app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert resp.status_code == 401


def test_valid_jwt_passes_middleware(tmp_path):
    app, provider = _oauth_app(tmp_path)
    access, _ = provider._mint_access_token("alice@example.com", ["patentkit"], "c")
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**MCP_HEADERS, "Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "patentkit"


# =========================================================================== #
# oauth-secret mode (self-contained, no external IdP, no network)             #
# =========================================================================== #


SECRET_VALUE = "open-sesame-correct-horse"


def _secret_env(tmp_path):
    return {
        "PATENTKIT_AUTH_MODE": "oauth-secret",
        "PATENTKIT_PUBLIC_URL": "https://patentkit.example.run.app",
        "PATENTKIT_OAUTH_SIGNING_SECRET": "super-secret-signing-key-at-least-32b!",
        "PATENTKIT_ACCESS_SECRET": SECRET_VALUE,
        "PATENTKIT_OAUTH_DIR": str(tmp_path / "oauth-secret"),
    }


def _secret_provider(tmp_path):
    config = load_oauth_config(_secret_env(tmp_path))
    return PatentkitSecretOAuthProvider(config)


# --------------------------------------------------------------------------- config


def test_secret_config_fails_fast_on_missing(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_oauth_config(
            {
                "PATENTKIT_AUTH_MODE": "oauth-secret",
                "PATENTKIT_PUBLIC_URL": "https://x.run.app",
            }
        )
    msg = str(exc.value)
    assert "PATENTKIT_OAUTH_SIGNING_SECRET" in msg
    assert "PATENTKIT_ACCESS_SECRET" in msg


def test_secret_config_needs_no_google_or_allowlist(tmp_path):
    config = load_oauth_config(_secret_env(tmp_path))
    assert config.mode == "oauth-secret"
    assert config.access_secret == SECRET_VALUE
    assert config.google_client_id == ""
    assert config.allowed_emails == frozenset()
    assert config.approve_url == "https://patentkit.example.run.app/auth/approve"
    assert config.resource_url == "https://patentkit.example.run.app/mcp"


# --------------------------------------------------------------------------- authorize


def test_secret_authorize_redirects_to_local_approve(tmp_path):
    provider = _secret_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()

    url = run(provider.authorize(client, _auth_params(challenge)))
    assert url.startswith("https://patentkit.example.run.app/auth/approve?")
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]
    assert state in provider._pending
    pending = provider._pending[state]
    assert pending.redirect_uri == "https://claude.ai/api/mcp/auth_callback"
    assert pending.code_challenge == challenge
    assert pending.client_state == "client-state-xyz"


# --------------------------------------------------------------------------- approve


def _secret_to_state(tmp_path):
    provider = _secret_provider(tmp_path)
    client = _client()
    run(provider.register_client(client))
    verifier, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]
    return provider, client, state, verifier, challenge


def test_secret_correct_code_issues_and_redirects(tmp_path):
    provider, client, state, _, _ = _secret_to_state(tmp_path)
    redirect, error = provider.complete_secret_approval(state, SECRET_VALUE)
    assert error is None
    from urllib.parse import parse_qs, urlparse

    parsed = parse_qs(urlparse(redirect).query)
    assert redirect.startswith("https://claude.ai/api/mcp/auth_callback?")
    assert parsed["state"] == ["client-state-xyz"]
    our_code = parsed["code"][0]
    assert our_code in provider._codes
    # The pending record is consumed exactly once.
    assert state not in provider._pending


def test_secret_wrong_code_issues_nothing(tmp_path):
    provider, client, state, _, _ = _secret_to_state(tmp_path)
    redirect, error = provider.complete_secret_approval(state, "wrong-code")
    assert error == "access_denied"
    assert redirect == ""
    assert provider._codes == {}
    # The pending record survives so the user can retry.
    assert state in provider._pending


def test_secret_unknown_state_rejected(tmp_path):
    provider = _secret_provider(tmp_path)
    redirect, error = provider.complete_secret_approval("no-such-state", SECRET_VALUE)
    assert error == "invalid_request"
    assert redirect == ""
    assert provider._codes == {}


def test_secret_empty_state_rejected(tmp_path):
    provider = _secret_provider(tmp_path)
    _, error = provider.complete_secret_approval(None, SECRET_VALUE)
    assert error == "invalid_request"


# --------------------------------------------------------------------------- token grant + PKCE


def _secret_full_to_code(tmp_path):
    provider, client, state, verifier, challenge = _secret_to_state(tmp_path)
    redirect, _ = provider.complete_secret_approval(state, SECRET_VALUE)
    from urllib.parse import parse_qs, urlparse

    our_code = parse_qs(urlparse(redirect).query)["code"][0]
    return provider, client, our_code, verifier


def test_secret_exchange_issues_jwt_with_fixed_subject(tmp_path):
    provider, client, code, _ = _secret_full_to_code(tmp_path)
    auth_code = run(provider.load_authorization_code(client, code))
    assert auth_code is not None
    assert auth_code.subject == SECRET_SUBJECT
    token = run(provider.exchange_authorization_code(client, auth_code))
    assert token.access_token and token.refresh_token
    assert token.token_type == "Bearer"

    access = run(provider.load_access_token(token.access_token))
    assert access is not None and access.subject == SECRET_SUBJECT
    assert "patentkit" in access.scopes


def test_secret_pkce_challenge_is_s256_of_verifier(tmp_path):
    # Mirror the Google test: the stored code_challenge equals S256(verifier),
    # so the SDK's upstream PKCE check is meaningful.
    provider, client, code, verifier = _secret_full_to_code(tmp_path)
    auth_code = run(provider.load_authorization_code(client, code))
    good = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert auth_code.code_challenge == good
    wrong = base64.urlsafe_b64encode(hashlib.sha256(b"not-the-verifier").digest()).decode().rstrip("=")
    assert wrong != auth_code.code_challenge


def test_secret_verify_token_accepts_fresh_rejects_tampered(tmp_path):
    provider, client, code, _ = _secret_full_to_code(tmp_path)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    tv = PatentkitTokenVerifier(provider)

    assert run(tv.verify_token(token.access_token)) is not None
    tampered = token.access_token[:-3] + ("aaa" if token.access_token[-3:] != "aaa" else "bbb")
    assert run(tv.verify_token(tampered)) is None
    assert run(tv.verify_token("not-a-jwt")) is None


def test_secret_verify_token_rejects_expired(tmp_path):
    provider = _secret_provider(tmp_path)
    import jwt

    now = int(time.time())
    payload = {
        "sub": SECRET_SUBJECT,
        "iss": provider.config.issuer,
        "aud": provider.config.resource_url,
        "iat": now - 100,
        "exp": now - 10,
        "scope": "patentkit",
        "client_id": "c",
        "jti": "x",
        "type": "access",
    }
    expired = jwt.encode(payload, provider.config.signing_secret, algorithm="HS256")
    assert run(provider.load_access_token(expired)) is None


def test_secret_refresh_rotates(tmp_path):
    provider, client, code, _ = _secret_full_to_code(tmp_path)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))

    rt = run(provider.load_refresh_token(client, token.refresh_token))
    assert rt is not None and rt.subject == SECRET_SUBJECT
    new = run(provider.exchange_refresh_token(client, rt, rt.scopes))
    assert new.access_token != token.access_token
    assert run(provider.load_access_token(new.access_token)) is not None
    # Old refresh token is denylisted after rotation.
    assert run(provider.load_refresh_token(client, token.refresh_token)) is None


def test_secret_revoke_denylists_access_token(tmp_path):
    provider, client, code, _ = _secret_full_to_code(tmp_path)
    auth_code = run(provider.load_authorization_code(client, code))
    token = run(provider.exchange_authorization_code(client, auth_code))
    access = run(provider.load_access_token(token.access_token))
    assert access is not None
    run(provider.revoke_token(access))
    assert run(provider.load_access_token(token.access_token)) is None


# --------------------------------------------------------------------------- app


def _secret_app(tmp_path):
    config = load_oauth_config(_secret_env(tmp_path))
    provider = PatentkitSecretOAuthProvider(config)
    app = build_oauth_http_app(PatentToolset(), config=config, provider=provider, json_response=True)
    return app, provider


def test_secret_metadata_advertises_endpoints(tmp_path):
    app, _ = _secret_app(tmp_path)
    with TestClient(app) as client:
        meta = client.get("/.well-known/oauth-authorization-server").json()
        assert meta["issuer"] == "https://patentkit.example.run.app/"
        assert meta["authorization_endpoint"].endswith("/authorize")
        assert meta["token_endpoint"].endswith("/token")
        assert meta["registration_endpoint"].endswith("/register")
        assert meta["revocation_endpoint"].endswith("/revoke")


def test_secret_approve_get_renders_form(tmp_path):
    app, provider = _secret_app(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]
    with TestClient(app) as tc:
        resp = tc.get(f"/auth/approve?state={state}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "<form" in body and 'type="password"' in body
        assert state in body  # hidden state field


def test_secret_approve_get_unknown_state_is_400(tmp_path):
    app, _ = _secret_app(tmp_path)
    with TestClient(app) as tc:
        resp = tc.get("/auth/approve?state=nope")
        assert resp.status_code == 400


def test_secret_approve_post_correct_then_wrong_via_http(tmp_path):
    app, provider = _secret_app(tmp_path)
    client = _client()
    run(provider.register_client(client))
    _, challenge = _pkce()
    url = run(provider.authorize(client, _auth_params(challenge)))
    from urllib.parse import parse_qs, urlparse

    state = parse_qs(urlparse(url).query)["state"][0]
    with TestClient(app) as tc:
        # Wrong secret: 401, re-rendered form, nothing issued.
        bad = tc.post(
            "/auth/approve",
            data={"state": state, "secret": "nope"},
            follow_redirects=False,
        )
        assert bad.status_code == 401
        assert "<form" in bad.text
        assert provider._codes == {}

        # Correct secret: 303 redirect back to claude.ai carrying our code.
        ok = tc.post(
            "/auth/approve",
            data={"state": state, "secret": SECRET_VALUE},
            follow_redirects=False,
        )
        assert ok.status_code == 303
        loc = ok.headers["location"]
        assert loc.startswith("https://claude.ai/api/mcp/auth_callback?")
        assert "code=" in loc and "state=client-state-xyz" in loc


def test_secret_unauthenticated_mcp_is_401(tmp_path):
    app, _ = _secret_app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert resp.status_code == 401


def test_secret_valid_jwt_passes_middleware(tmp_path):
    app, provider = _secret_app(tmp_path)
    access, _ = provider._mint_access_token(SECRET_SUBJECT, ["patentkit"], "c")
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**MCP_HEADERS, "Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "patentkit"
