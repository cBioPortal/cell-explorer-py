"""End-to-end test: API mints a token, zarr-server validates and serves."""

import time
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
from fastapi.testclient import TestClient

from zarr_auth_proxy.config import Settings
from zarr_auth_proxy.main import create_app


@pytest.fixture()
def e2e_setup(tmp_path):
    """Set up key pair, data dir, and both the signing and verifying sides."""
    # Generate key pair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

    public_file = tmp_path / "public.pem"
    public_file.write_bytes(public_pem)

    # Create test data
    data_dir = tmp_path / "data"
    zarr_dir = data_dir / "mydata" / "atlas.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text('{"zarr_format": 3}')
    chunk_dir = zarr_dir / "X" / "c" / "0"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "0").write_bytes(b"chunk-data-here")

    # Create zarr-server app
    settings = Settings(public_key_file=public_file, data_dir=data_dir)
    app = create_app(settings)
    client = TestClient(app)

    return {
        "client": client,
        "private_pem": private_pem,
        "public_pem": public_pem,
    }


def _mint_like_api(private_pem, path, ttl=1800):
    """Simulate what the API's _mint_http_token does."""
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl)
    token = jwt.encode(
        {
            "path": path,
            "datasource": "test-datasource-id",
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
        },
        private_pem,
        algorithm="RS256",
    )
    return token


def test_e2e_valid_token_serves_file(e2e_setup):
    client = e2e_setup["client"]
    token = _mint_like_api(e2e_setup["private_pem"], "mydata/atlas.zarr")

    response = client.get(
        "/mydata/atlas.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "zarr_format" in response.text


def test_e2e_valid_token_serves_chunk(e2e_setup):
    client = e2e_setup["client"]
    token = _mint_like_api(e2e_setup["private_pem"], "mydata/atlas.zarr")

    response = client.get(
        "/mydata/atlas.zarr/X/c/0/0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content == b"chunk-data-here"


def test_e2e_wrong_path_rejected(e2e_setup):
    client = e2e_setup["client"]
    token = _mint_like_api(e2e_setup["private_pem"], "mydata/other.zarr")

    response = client.get(
        "/mydata/atlas.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_e2e_different_key_rejected(e2e_setup):
    """Token signed with a different private key is rejected."""
    client = e2e_setup["client"]
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    token = _mint_like_api(other_pem, "mydata/atlas.zarr")

    response = client.get(
        "/mydata/atlas.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401
