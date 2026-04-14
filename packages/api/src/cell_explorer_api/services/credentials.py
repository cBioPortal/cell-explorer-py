"""Credential minting service for datasource-specific access tokens."""

import os
from datetime import datetime, timezone, timedelta

import jwt

from cell_explorer_api.db.models import Datasource, DatasourceType

DEFAULT_TTL_SECONDS = 1800  # 30 minutes


class CredentialError(Exception):
    """Raised when credentials cannot be minted."""


def mint_credentials(
    datasource: Datasource,
    path: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict:
    """Mint short-lived credentials for accessing a dataset.

    Returns a dict with credential_type, url, and type-specific fields.
    Raises CredentialError if credentials cannot be minted.
    """
    if datasource.credential_ref is None:
        raise CredentialError(f"No credential_ref configured for datasource '{datasource.name}'")

    url = f"{datasource.base_url}/{path}"
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    minters = {
        DatasourceType.S3_CLOUDFRONT: _mint_cloudfront,
        DatasourceType.HTTP_TOKEN: _mint_http_token,
    }

    minter = minters.get(datasource.type)
    if minter is None:
        raise CredentialError(f"Unsupported datasource type: {datasource.type}")

    return minter(datasource, path, url, expires_at, ttl_seconds)


def _mint_http_token(
    datasource: Datasource,
    path: str,
    url: str,
    expires_at: datetime,
    ttl_seconds: int,
) -> dict:
    """Mint a signed RS256 JWT for HTTP token-based access."""
    env_key = f"DATASOURCE_{datasource.credential_ref}_PRIVATE_KEY_FILE"
    key_path = os.environ.get(env_key)
    if not key_path:
        raise CredentialError(
            f"Credentials not configured: {env_key} environment variable is not set"
        )

    from pathlib import Path

    key_file = Path(key_path)
    if not key_file.is_file():
        raise CredentialError(
            f"Private key file not found: {key_path}"
        )

    private_key = key_file.read_bytes()

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "path": path,
            "datasource": str(datasource.id),
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
        },
        private_key,
        algorithm="RS256",
    )

    return {
        "url": url,
        "credential_type": "bearer_token",
        "token": token,
        "expires_at": expires_at.isoformat(),
    }


def _mint_cloudfront(
    datasource: Datasource,
    path: str,
    url: str,
    expires_at: datetime,
    ttl_seconds: int,
) -> dict:
    """Mint CloudFront signed cookies.

    Note: Full CloudFront signing requires the cryptography library's
    RSA signing. This is a placeholder structure — the actual signing
    logic will be implemented when CloudFront integration is tested.
    """
    key_pair_id_key = f"DATASOURCE_{datasource.credential_ref}_KEY_PAIR_ID"
    private_key_key = f"DATASOURCE_{datasource.credential_ref}_PRIVATE_KEY"

    key_pair_id = os.environ.get(key_pair_id_key)
    private_key = os.environ.get(private_key_key)

    if not key_pair_id or not private_key:
        missing = []
        if not key_pair_id:
            missing.append(key_pair_id_key)
        if not private_key:
            missing.append(private_key_key)
        raise CredentialError(
            f"Credentials not configured: {', '.join(missing)} environment variable(s) not set"
        )

    # CloudFront signed cookie generation will be implemented
    # when integration testing with a real CloudFront distribution.
    # For now, return the structure so the API contract is established.
    return {
        "url": url,
        "credential_type": "signed_cookies",
        "cookies": {
            "CloudFront-Policy": "TODO",
            "CloudFront-Signature": "TODO",
            "CloudFront-Key-Pair-Id": key_pair_id,
        },
        "expires_at": expires_at.isoformat(),
    }
