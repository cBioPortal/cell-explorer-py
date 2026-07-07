"""Tests for credential minting service."""

import base64
import json
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from cell_explorer_api.db.models import Datasource, DatasourceType
from cell_explorer_api.services.credentials import (
    CredentialError,
    mint_credentials,
)


@pytest.fixture()
def rsa_key_file(tmp_path):
    """Generate an RSA private key and write to a temp file."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    key_file = tmp_path / "private.pem"
    key_file.write_bytes(private_pem)
    return key_file, public_pem


@pytest.fixture()
def http_token_datasource():
    return Datasource(
        name="Lab Server",
        type=DatasourceType.HTTP_TOKEN,
        base_url="https://lab.example.com",
        credential_ref="LAB_SERVER",
    )


def test_mint_http_token_credentials(http_token_datasource, rsa_key_file, monkeypatch):
    key_file, public_pem = rsa_key_file
    monkeypatch.setenv("DATASOURCE_LAB_SERVER_PRIVATE_KEY_FILE", str(key_file))
    result = mint_credentials(http_token_datasource, "datasets/test.zarr")
    assert result["credential_type"] == "bearer_token"
    assert result["url"] == "https://lab.example.com/datasets/test.zarr"
    assert "token" in result
    assert "expires_at" in result
    # Verify token is valid RS256 JWT
    decoded = jwt.decode(result["token"], public_pem, algorithms=["RS256"])
    assert decoded["path"] == "datasets/test.zarr"


def test_mint_credentials_missing_key_file(http_token_datasource):
    with pytest.raises(CredentialError, match="not configured"):
        mint_credentials(http_token_datasource, "datasets/test.zarr")


def test_mint_credentials_key_file_not_found(http_token_datasource, monkeypatch):
    monkeypatch.setenv("DATASOURCE_LAB_SERVER_PRIVATE_KEY_FILE", "/nonexistent/key.pem")
    with pytest.raises(CredentialError, match="not found"):
        mint_credentials(http_token_datasource, "datasets/test.zarr")


def test_mint_credentials_no_credential_ref():
    ds = Datasource(
        name="Public CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://public.cdn.net",
        credential_ref=None,
    )
    with pytest.raises(CredentialError, match="No credential_ref"):
        mint_credentials(ds, "datasets/test.zarr")


def _cf_b64_decode(s: str) -> bytes:
    """Reverse CloudFront's URL-safe base64 (test helper)."""
    return base64.b64decode(s.replace("-", "+").replace("_", "=").replace("~", "/"))


@pytest.fixture()
def cloudfront_keypair():
    """RSA private key PEM (str) + public key object for signature verification."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    return private_pem, key.public_key()


@pytest.fixture()
def cloudfront_datasource():
    return Datasource(
        name="Protected CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://cf.example.com",
        credential_ref="TEST",
    )


def test_mint_cloudfront_signed_cookies(cloudfront_datasource, cloudfront_keypair, monkeypatch):
    private_pem, public_key = cloudfront_keypair
    monkeypatch.setenv("DATASOURCE_TEST_KEY_PAIR_ID", "K2XYZ123")
    monkeypatch.setenv("DATASOURCE_TEST_PRIVATE_KEY", private_pem)

    result = mint_credentials(cloudfront_datasource, "protected/spectrum.zarr")

    assert result["credential_type"] == "signed_cookies"
    assert result["url"] == "https://cf.example.com/protected/spectrum.zarr"
    cookies = result["cookies"]
    assert cookies["CloudFront-Key-Pair-Id"] == "K2XYZ123"
    assert cookies["CloudFront-Policy"] != "TODO"
    assert cookies["CloudFront-Signature"] != "TODO"

    policy_bytes = _cf_b64_decode(cookies["CloudFront-Policy"])
    sig_bytes = _cf_b64_decode(cookies["CloudFront-Signature"])

    # The load-bearing assertion: CloudFront would accept this (RSA-SHA1 over the policy).
    public_key.verify(sig_bytes, policy_bytes, padding.PKCS1v15(), hashes.SHA1())

    policy = json.loads(policy_bytes)
    stmt = policy["Statement"][0]
    assert stmt["Resource"] == "https://cf.example.com/protected/spectrum.zarr/*"
    assert isinstance(stmt["Condition"]["DateLessThan"]["AWS:EpochTime"], int)


def test_mint_cloudfront_missing_env(cloudfront_datasource):
    with pytest.raises(CredentialError, match="not set"):
        mint_credentials(cloudfront_datasource, "protected/spectrum.zarr")


def test_mint_cloudfront_bad_private_key(cloudfront_datasource, monkeypatch):
    monkeypatch.setenv("DATASOURCE_TEST_KEY_PAIR_ID", "K2XYZ123")
    monkeypatch.setenv("DATASOURCE_TEST_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----")
    with pytest.raises(CredentialError, match="[Ii]nvalid private key"):
        mint_credentials(cloudfront_datasource, "protected/spectrum.zarr")
