"""Keycloak OIDC client — token validation, auth URLs, token exchange."""

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


class KeycloakClient:
    """Handles Keycloak OIDC operations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = f"{settings.keycloak_url}/realms/{settings.keycloak_realm}"
        self._oidc = f"{self._base}/protocol/openid-connect"
        self._jwks: dict[str, bytes] = {}  # kid → PEM public key bytes

    def authorization_url(self, redirect_uri: str, state: str) -> str:
        """Build the Keycloak authorization endpoint URL."""
        params = {
            "client_id": self._settings.keycloak_client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": "openid profile email",
        }
        return f"{self._oidc}/auth?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        """Exchange an authorization code for tokens."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._oidc}/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._settings.keycloak_client_id,
                    "client_secret": self._settings.keycloak_client_secret,
                },
            )
            response.raise_for_status()
            return response.json()

    async def refresh_token(self, refresh_token: str) -> dict:
        """Use a refresh token to get a new access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._oidc}/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self._settings.keycloak_client_id,
                    "client_secret": self._settings.keycloak_client_secret,
                },
            )
            response.raise_for_status()
            return response.json()

    async def fetch_jwks(self) -> None:
        """Fetch and cache Keycloak's JSON Web Key Set."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._oidc}/certs")
            response.raise_for_status()
            jwks = response.json()

        self._jwks = {}
        for key_data in jwks.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
                self._jwks[kid] = public_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo,
                )

    def decode_token(self, token: str) -> User:
        """Decode and validate a JWT access token. Raises on invalid/expired."""
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if kid not in self._jwks:
            raise jwt.InvalidTokenError(f"Unknown kid: {kid}")

        public_key = load_pem_public_key(self._jwks[kid])
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=self._settings.oidc_issuer_url,
            options={"verify_aud": False},
        )

        # Keycloak sets aud="account" by default; verify azp (authorized party) instead
        azp = claims.get("azp")
        if azp != self._settings.keycloak_client_id:
            raise jwt.InvalidTokenError(
                f"Authorized party mismatch: expected {self._settings.keycloak_client_id}, got {azp}"
            )

        realm_roles = claims.get("realm_access", {}).get("roles", [])
        return User(
            sub=claims["sub"],
            name=claims.get("name"),
            email=claims.get("email"),
            roles=realm_roles,
        )

    def logout_url(self, redirect_uri: str) -> str:
        """Build the Keycloak logout endpoint URL."""
        params = {
            "client_id": self._settings.keycloak_client_id,
            "post_logout_redirect_uri": redirect_uri,
        }
        return f"{self._oidc}/logout?{urlencode(params)}"
