"""Tests for the generic OIDC client."""

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from cell_explorer_api.auth.oidc import OidcClient, extract_roles
from cell_explorer_api.config import Settings

DISCOVERY = {
    "issuer": "https://idp.example.com",
    "authorization_endpoint": "https://idp.example.com/authorize",
    "token_endpoint": "https://idp.example.com/token",
    "jwks_uri": "https://idp.example.com/jwks",
    "end_session_endpoint": "https://idp.example.com/logout",
}


def _keys():
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return pk, pk.public_key()


def _client(settings: Settings, public_key=None, kid="kid1") -> OidcClient:
    c = OidcClient(settings)
    c._apply_discovery(DISCOVERY)
    if public_key is not None:
        c._jwks = {kid: public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)}
    return c


def _entra_settings(**o):
    return Settings(auth_provider="entra", oidc_issuer="https://idp.example.com",
                    oidc_client_id="app", oidc_client_secret="s", **o)


def _kc_settings(**o):
    return Settings(keycloak_url="https://idp.example.com", keycloak_realm="r",
                    keycloak_client_id="app", keycloak_client_secret="s", **o)


class TestExtractRoles:
    def test_keycloak_merges_realm_and_client_roles(self):
        claims = {
            "realm_access": {"roles": ["viewer", "admin"]},
            "resource_access": {"app": {"roles": ["curator"]}},
        }
        roles = extract_roles(claims, ["realm_access.roles", "resource_access.app.roles"])
        assert roles == ["admin", "curator", "viewer"]

    def test_entra_single_roles_claim(self):
        assert extract_roles({"roles": ["viewer"]}, ["roles"]) == ["viewer"]

    def test_missing_path_contributes_nothing(self):
        assert extract_roles({"roles": ["viewer"]}, ["nope.here", "roles"]) == ["viewer"]

    def test_no_paths_yields_no_roles(self):
        assert extract_roles({"roles": ["viewer"]}, []) == []


class TestAuthorizationUrl:
    def test_includes_scopes_and_no_idp_hint_for_entra(self):
        url = _client(_entra_settings()).authorization_url("https://cb/cb", "st8")
        assert url.startswith("https://idp.example.com/authorize?")
        assert "scope=openid+profile+email+offline_access" in url
        assert "kc_idp_hint" not in url
        assert "state=st8" in url and "response_type=code" in url

    def test_keycloak_idp_hint_included(self):
        url = _client(_kc_settings(keycloak_idp_hint="pingId")).authorization_url("https://cb", "s")
        assert "kc_idp_hint=pingId" in url


class TestDecodeToken:
    def test_entra_roles_and_aud_iss(self):
        pk, pub = _keys()
        c = _client(_entra_settings(), public_key=pub)
        token = jwt.encode(
            {"sub": "u1", "name": "N", "email": "e@x", "roles": ["viewer"],
             "aud": "app", "iss": "https://idp.example.com"},
            pk, algorithm="RS256", headers={"kid": "kid1"},
        )
        user = c.decode_token(token)
        assert user.sub == "u1" and user.roles == ["viewer"]

    def test_wrong_audience_rejected(self):
        pk, pub = _keys()
        c = _client(_entra_settings(), public_key=pub)
        token = jwt.encode(
            {"sub": "u", "aud": "WRONG", "iss": "https://idp.example.com"},
            pk, algorithm="RS256", headers={"kid": "kid1"},
        )
        with pytest.raises(jwt.InvalidAudienceError):
            c.decode_token(token)

    def test_wrong_issuer_rejected(self):
        pk, pub = _keys()
        c = _client(_entra_settings(), public_key=pub)
        token = jwt.encode(
            {"sub": "u", "aud": "app", "iss": "https://attacker.example.com"},
            pk, algorithm="RS256", headers={"kid": "kid1"},
        )
        with pytest.raises(jwt.InvalidIssuerError):
            c.decode_token(token)

    def test_keycloak_dual_role_merge(self):
        pk, pub = _keys()
        c = _client(_kc_settings(), public_key=pub)
        token = jwt.encode(
            {"sub": "u", "aud": "app", "iss": "https://idp.example.com",
             "realm_access": {"roles": ["viewer"]},
             "resource_access": {"app": {"roles": ["curator"]}}},
            pk, algorithm="RS256", headers={"kid": "kid1"},
        )
        assert c.decode_token(token).roles == ["curator", "viewer"]


class TestDiscoveryParsing:
    def test_apply_discovery_sets_endpoints(self):
        c = OidcClient(_entra_settings())
        c._apply_discovery(DISCOVERY)
        assert c._authorization_endpoint == "https://idp.example.com/authorize"
        assert c._token_endpoint == "https://idp.example.com/token"
        assert c._jwks_uri == "https://idp.example.com/jwks"
        assert c._issuer == "https://idp.example.com"
        assert c._end_session_endpoint == "https://idp.example.com/logout"


class TestLogoutUrl:
    def test_uses_end_session_endpoint(self):
        url = _client(_entra_settings()).logout_url("https://cb/home")
        assert url.startswith("https://idp.example.com/logout?")
        assert "post_logout_redirect_uri=https%3A%2F%2Fcb%2Fhome" in url


class TestAuthorizationUrlPrompt:
    def test_no_prompt_by_default(self):
        url = _client(_kc_settings()).authorization_url("https://cb/cb", "st")
        assert "prompt=" not in url

    def test_prompt_is_passed_through(self):
        url = _client(_kc_settings()).authorization_url("https://cb/cb", "st", prompt="select_account")
        assert "prompt=select_account" in url


def _capture_posts(monkeypatch, status_code=204):
    """Route the OIDC client's HTTP calls through a MockTransport; return the captured requests."""
    import httpx

    from cell_explorer_api.auth import oidc as oidc_module

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        oidc_module.httpx, "AsyncClient",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(handler)),
    )
    return seen


class TestEndSession:
    async def test_keycloak_posts_refresh_token_to_end_session_endpoint(self, monkeypatch):
        from urllib.parse import parse_qs

        seen = _capture_posts(monkeypatch)
        await _client(_kc_settings()).end_session("the-refresh-token")
        assert len(seen) == 1
        assert str(seen[0].url) == "https://idp.example.com/logout"
        assert seen[0].method == "POST"
        form = parse_qs(seen[0].content.decode())
        assert form == {
            "client_id": ["app"],
            "client_secret": ["s"],
            "refresh_token": ["the-refresh-token"],
        }

    async def test_keycloak_raises_when_rejected(self, monkeypatch):
        import httpx

        _capture_posts(monkeypatch, status_code=400)
        with pytest.raises(httpx.HTTPStatusError):
            await _client(_kc_settings()).end_session("bad")

    async def test_entra_makes_no_request(self, monkeypatch):
        # Entra has no back-channel logout; signing out there relies on the
        # account-chooser prompt at the next sign-in instead.
        seen = _capture_posts(monkeypatch)
        await _client(_entra_settings()).end_session("the-refresh-token")
        assert seen == []
