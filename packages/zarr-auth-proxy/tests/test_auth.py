"""Tests for JWT validation and path checking."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from zarr_auth_proxy.auth import validate_token, is_path_authorized


@pytest.fixture()
def rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem


def _sign(private_pem, path, ttl=300):
    return jwt.encode(
        {
            "path": path,
            "datasource": "test-ds-id",
            "exp": int(time.time()) + ttl,
            "iat": int(time.time()),
        },
        private_pem,
        algorithm="RS256",
    )


def test_validate_token_valid(rsa_keys):
    private_pem, public_pem = rsa_keys
    token = _sign(private_pem, "datasets/brca.zarr")
    claims = validate_token(token, public_pem)
    assert claims["path"] == "datasets/brca.zarr"
    assert claims["datasource"] == "test-ds-id"


def test_validate_token_expired(rsa_keys):
    private_pem, public_pem = rsa_keys
    token = _sign(private_pem, "datasets/brca.zarr", ttl=-10)
    assert validate_token(token, public_pem) is None


def test_validate_token_invalid_signature(rsa_keys):
    _, public_pem = rsa_keys
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    token = _sign(other_pem, "datasets/brca.zarr")
    assert validate_token(token, public_pem) is None


def test_validate_token_garbage():
    assert validate_token("not-a-jwt", b"not-a-key") is None


# --- Path authorization ---


def test_path_authorized_exact_prefix():
    assert is_path_authorized("datasets/brca.zarr/zarr.json", "datasets/brca.zarr") is True


def test_path_authorized_nested():
    assert is_path_authorized("datasets/brca.zarr/X/c/0/0", "datasets/brca.zarr") is True


def test_path_authorized_mismatch():
    assert is_path_authorized("datasets/pdx/zarr.json", "datasets/brca.zarr") is False


def test_path_authorized_boundary_attack():
    """brca.zarr.evil should not match brca.zarr."""
    assert is_path_authorized("datasets/brca.zarr.evil/data", "datasets/brca.zarr") is False


def test_path_authorized_exact_match():
    """Requesting exactly the token path (no trailing slash or subpath) is allowed."""
    assert is_path_authorized("datasets/brca.zarr", "datasets/brca.zarr") is True


def test_path_authorized_with_trailing_slash():
    assert is_path_authorized("datasets/brca.zarr/", "datasets/brca.zarr") is True
