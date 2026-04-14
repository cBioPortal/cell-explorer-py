"""Tests for credential minting service."""

from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
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
