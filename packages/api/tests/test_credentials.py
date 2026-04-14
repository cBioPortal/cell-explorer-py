"""Tests for credential minting service."""

import os
import time

import jwt
import pytest

from cell_explorer_api.db.models import Datasource, DatasourceType
from cell_explorer_api.services.credentials import (
    CredentialError,
    mint_credentials,
)


@pytest.fixture()
def http_token_datasource():
    return Datasource(
        name="Lab Server",
        type=DatasourceType.HTTP_TOKEN,
        base_url="https://lab.example.com",
        credential_ref="LAB_SERVER",
    )


def test_mint_http_token_credentials(http_token_datasource, monkeypatch):
    monkeypatch.setenv("DATASOURCE_LAB_SERVER_SIGNING_SECRET", "test-secret-key")
    result = mint_credentials(http_token_datasource, "datasets/test.zarr")
    assert result["credential_type"] == "bearer_token"
    assert result["url"] == "https://lab.example.com/datasets/test.zarr"
    assert "token" in result
    assert "expires_at" in result
    # Verify token is valid JWT
    decoded = jwt.decode(result["token"], "test-secret-key", algorithms=["HS256"])
    assert decoded["path"] == "datasets/test.zarr"


def test_mint_credentials_missing_env_var(http_token_datasource):
    # Don't set the env var
    with pytest.raises(CredentialError, match="not configured"):
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
