"""Generic OIDC client — discovery, token validation, auth URLs, token exchange.

Works with any OIDC provider (Keycloak, Microsoft Entra, ...). Endpoints are
resolved from the provider's .well-known/openid-configuration; roles are read
from configurable dotted claim-paths. Provider presets live in Settings.
"""

import logging
from urllib.parse import urlencode

import httpx
import jwt
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)

from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings

logger = logging.getLogger(__name__)


def extract_roles(claims: dict, paths: list[str]) -> list[str]:
    """Merge role lists found at each dotted JSON-path into a sorted, deduped list."""
    roles: set[str] = set()
    for path in paths:
        node = claims
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, list):
            roles.update(str(r) for r in node)
    return sorted(roles)


class OidcClient:
    """Provider-agnostic OIDC operations, driven by resolved Settings + discovery."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jwks: dict[str, bytes] = {}  # kid → PEM public key bytes
        self._authorization_endpoint: str | None = None
        self._token_endpoint: str | None = None
        self._jwks_uri: str | None = None
        self._issuer: str | None = None
        self._end_session_endpoint: str | None = None

    def _apply_discovery(self, doc: dict) -> None:
        self._authorization_endpoint = doc.get("authorization_endpoint")
        self._token_endpoint = doc.get("token_endpoint")
        self._jwks_uri = doc.get("jwks_uri")
        self._issuer = doc.get("issuer")
        self._end_session_endpoint = doc.get("end_session_endpoint")

    async def discover(self) -> None:
        """Fetch .well-known/openid-configuration and cache the endpoints."""
        url = self._settings.discovery_url
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            self._apply_discovery(response.json())

    def authorization_url(self, redirect_uri: str, state: str) -> str:
        params = {
            "client_id": self._settings.resolved_client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": self._settings.resolved_scopes,
        }
        if self._settings.resolved_idp_hint:
            params["kc_idp_hint"] = self._settings.resolved_idp_hint
        return f"{self._authorization_endpoint}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._settings.resolved_client_id,
                    "client_secret": self._settings.resolved_client_secret,
                },
            )
            response.raise_for_status()
            return response.json()

    async def refresh_token(self, refresh_token: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self._settings.resolved_client_id,
                    "client_secret": self._settings.resolved_client_secret,
                },
            )
            response.raise_for_status()
            return response.json()

    async def fetch_jwks(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(self._jwks_uri)
            response.raise_for_status()
            jwks = response.json()
        self._jwks = {}
        for key_data in jwks.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
                self._jwks[kid] = public_key.public_bytes(
                    encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo,
                )

    def decode_token(self, token: str) -> User:
        """Decode and validate a JWT access token. Raises on invalid/expired."""
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if kid not in self._jwks:
            raise jwt.InvalidTokenError(f"Unknown kid: {kid}")

        public_key = load_pem_public_key(self._jwks[kid])
        # leeway=30s tolerates small clock skew between the IdP and this
        # container (see issue #131).
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=self._settings.resolved_audience,
            issuer=self._issuer or self._settings.resolved_issuer,
            leeway=30,
        )
        return User(
            sub=claims["sub"],
            name=claims.get("name"),
            email=claims.get("email"),
            roles=extract_roles(claims, self._settings.resolved_roles_claims),
        )

    def logout_url(self, redirect_uri: str) -> str:
        params = {
            "client_id": self._settings.resolved_client_id,
            "post_logout_redirect_uri": redirect_uri,
        }
        return f"{self._end_session_endpoint}?{urlencode(params)}"
