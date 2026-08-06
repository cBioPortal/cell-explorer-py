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
