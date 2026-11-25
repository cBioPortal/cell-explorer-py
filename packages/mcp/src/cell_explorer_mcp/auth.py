"""Signed-in tier: verify user tokens by exchanging them at the Cell Explorer API."""

import hashlib
import logging
import time

import httpx
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token

logger = logging.getLogger(__name__)

GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
EXPIRY_MARGIN_S = 30
MAX_CACHE_ENTRIES = 2000


class CellExplorerTokenVerifier(TokenVerifier):
    """Trades the user's token plus the MCP key for a short-lived API token."""

    def __init__(
        self,
        *,
        exchange_url: str,
        mcp_key: str,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__()
        self._url = exchange_url
        self._mcp_key = mcp_key
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._cache: dict[str, AccessToken] = {}

    async def verify_token(self, token: str) -> AccessToken | None:
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None and cached.expires_at - EXPIRY_MARGIN_S > time.time():
            return cached
        try:
            resp = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._mcp_key}"},
                data={
                    "grant_type": GRANT_TYPE,
                    "subject_token": token,
                    "subject_token_type": TOKEN_TYPE,
                },
            )
        except httpx.HTTPError as e:
            logger.warning("token exchange failed: %s", type(e).__name__)
            return None
        if resp.status_code != 200:
            if resp.status_code not in (400, 401):
                logger.warning("token exchange returned %s", resp.status_code)
            return None
        try:
            body = resp.json()
            access = AccessToken(
                token=token,
                client_id="cell-explorer",
                scopes=[],
                expires_at=int(time.time() + body["expires_in"]),
                claims={"api_token": body["access_token"], "sub": body["subject"]},
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("malformed token exchange response: %s", type(e).__name__)
            return None
        self._store(key, access)
        return access

    def _store(self, key: str, access: AccessToken) -> None:
        if key not in self._cache and len(self._cache) >= MAX_CACHE_ENTRIES:
            now = time.time()
            self._cache = {
                k: v for k, v in self._cache.items() if v.expires_at > now
            }
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = access


def _claim(name: str) -> str:
    access = get_access_token()
    if access is None or name not in access.claims:
        raise RuntimeError("no authenticated user for this request")
    return access.claims[name]


def current_api_token() -> str:
    return _claim("api_token")


def current_user_sub() -> str:
    return _claim("sub")
