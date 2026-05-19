"""Tests for Keycloak OIDC client."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.config import Settings


def _generate_rsa_keypair():
    """Generate an RSA key pair for testing."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_settings(**overrides) -> Settings:
    defaults = {
        "keycloak_url": "https://auth.example.com",
        "keycloak_realm": "test-realm",
        "keycloak_client_id": "test-client",
        "keycloak_client_secret": "test-secret",
    }
    return Settings(**(defaults | overrides))


def _encode_token(claims: dict, private_key, kid: str = "test-kid") -> str:
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture()
def rsa_keys():
    return _generate_rsa_keypair()


@pytest.fixture()
def keycloak(rsa_keys):
    settings = _make_settings()
    client = KeycloakClient(settings)
    _, public_key = rsa_keys
    client._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    return client


def test_authorization_url():
    settings = _make_settings()
    client = KeycloakClient(settings)
    url = client.authorization_url(
        redirect_uri="https://app.example.com/api/auth/callback",
        state="abc123",
    )
    assert "https://auth.example.com/realms/test-realm/protocol/openid-connect/auth" in url
    assert "client_id=test-client" in url
    assert "redirect_uri=" in url
    assert "state=abc123" in url
    assert "response_type=code" in url
    assert "kc_idp_hint" not in url


def test_authorization_url_with_idp_hint():
    settings = _make_settings(keycloak_idp_hint="pingId")
    client = KeycloakClient(settings)
    url = client.authorization_url(
        redirect_uri="https://app.example.com/api/auth/callback",
        state="abc123",
    )
    assert "kc_idp_hint=pingId" in url


def test_decode_valid_token(keycloak, rsa_keys):
    private_key, _ = rsa_keys
    claims = {
        "sub": "user-123",
        "name": "Test User",
        "email": "test@example.com",
        "realm_access": {"roles": ["viewer"]},
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    }
    token = _encode_token(claims, private_key)
    user = keycloak.decode_token(token)
    assert user.sub == "user-123"
    assert user.name == "Test User"
    assert user.email == "test@example.com"
    assert user.roles == ["viewer"]


def test_decode_merges_realm_and_client_roles(keycloak, rsa_keys):
    private_key, _ = rsa_keys
    claims = {
        "sub": "user-123",
        "realm_access": {"roles": ["realm-role", "shared"]},
        "resource_access": {
            "test-client": {"roles": ["client-role", "shared"]},
            "other-client": {"roles": ["should-not-include"]},
        },
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    }
    token = _encode_token(claims, private_key)
    user = keycloak.decode_token(token)
    # Merged, deduplicated, sorted
    assert user.roles == ["client-role", "realm-role", "shared"]
    assert "should-not-include" not in user.roles


def test_decode_expired_token_raises(keycloak, rsa_keys):
    private_key, _ = rsa_keys
    claims = {
        "sub": "user-123",
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        # Past the 30s leeway window so the decoder genuinely rejects.
        "exp": int(time.time()) - 60,
        "iat": int(time.time()) - 300,
    }
    token = _encode_token(claims, private_key)
    with pytest.raises(jwt.ExpiredSignatureError):
        keycloak.decode_token(token)


def test_decode_invalid_signature_raises(keycloak):
    other_key, _ = _generate_rsa_keypair()
    claims = {
        "sub": "user-123",
        "iss": "https://auth.example.com/realms/test-realm",
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    }
    token = _encode_token(claims, other_key)
    with pytest.raises(jwt.InvalidSignatureError):
        keycloak.decode_token(token)
