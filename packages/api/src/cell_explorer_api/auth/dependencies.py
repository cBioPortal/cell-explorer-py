"""FastAPI auth dependencies."""

import logging

from fastapi import HTTPException, Request

from cell_explorer_api.auth.oidc import OidcClient
from cell_explorer_api.auth.models import User

logger = logging.getLogger(__name__)


async def require_auth(request: Request) -> User:
    """Validate session cookie and return User. Raises 401 if not authenticated."""
    access_token = request.cookies.get("cce_access")
    refresh_token = request.cookies.get("cce_refresh")

    # No credentials at all — short-circuit before hitting OIDC config.
    if not access_token and not refresh_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not request.app.state.settings.auth_enabled:
        raise HTTPException(status_code=501, detail="Authentication is not configured")

    oidc: OidcClient = request.app.state.oidc

    # Happy path: try to decode the access token if we have one.
    if access_token:
        try:
            return oidc.decode_token(access_token)
        except Exception as e:
            logger.warning("Access token decode failed: %s", e)

    # Access token missing or invalid — fall back to refresh.
    if not refresh_token:
        # Access cookie was present but invalid, and no refresh cookie — user must log in.
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        logger.info("Attempting token refresh")
        tokens = await oidc.refresh_token(refresh_token)
        user = oidc.decode_token(tokens["access_token"])
        request.state.new_access_token = tokens["access_token"]
        request.state.new_refresh_token = tokens.get("refresh_token", refresh_token)
        return user
    except Exception as refresh_err:
        logger.warning("Token refresh failed: %s", refresh_err)
        raise HTTPException(status_code=401, detail="Session expired")
