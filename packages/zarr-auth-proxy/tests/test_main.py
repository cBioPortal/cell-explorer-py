"""Tests for zarr auth proxy FastAPI app."""

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

from zarr_auth_proxy.main import create_app
from zarr_auth_proxy.config import Settings


@pytest.fixture()
def rsa_keys(tmp_path):
    """Generate RSA key pair and write to files."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

    private_file = tmp_path / "private.pem"
    public_file = tmp_path / "public.pem"
    private_file.write_bytes(private_pem)
    public_file.write_bytes(public_pem)

    return private_pem, public_file


@pytest.fixture()
def data_dir(tmp_path):
    """Create a test data directory with zarr-like files."""
    zarr_dir = tmp_path / "data" / "datasets" / "test.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text('{"zarr_format": 3}')

    chunk_dir = zarr_dir / "X" / "c" / "0"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "0").write_bytes(b"\x00\x01\x02\x03")

    return tmp_path / "data"


@pytest.fixture()
def client(rsa_keys, data_dir):
    _, public_file = rsa_keys
    settings = Settings(public_key_file=public_file, data_dir=data_dir)
    app = create_app(settings)
    return TestClient(app)


@pytest.fixture()
def private_pem(rsa_keys):
    return rsa_keys[0]


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


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_serve_file_with_valid_token(client, private_pem):
    token = _sign(private_pem, "datasets/test.zarr")
    response = client.get(
        "/datasets/test.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "zarr_format" in response.text


def test_serve_binary_chunk(client, private_pem):
    token = _sign(private_pem, "datasets/test.zarr")
    response = client.get(
        "/datasets/test.zarr/X/c/0/0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.content == b"\x00\x01\x02\x03"


def test_missing_auth_header(client):
    response = client.get("/datasets/test.zarr/zarr.json")
    assert response.status_code == 401


def test_invalid_token(client):
    response = client.get(
        "/datasets/test.zarr/zarr.json",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_expired_token(client, private_pem):
    token = _sign(private_pem, "datasets/test.zarr", ttl=-10)
    response = client.get(
        "/datasets/test.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_path_not_authorized(client, private_pem):
    token = _sign(private_pem, "datasets/other.zarr")
    response = client.get(
        "/datasets/test.zarr/zarr.json",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_file_not_found(client, private_pem):
    token = _sign(private_pem, "datasets/test.zarr")
    response = client.get(
        "/datasets/test.zarr/nonexistent",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


def test_directory_traversal_blocked(client, private_pem):
    token = _sign(private_pem, "datasets/test.zarr")
    response = client.get(
        "/datasets/test.zarr/../../etc/passwd",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (401, 404)
