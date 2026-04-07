"""FastAPI auth dependencies."""

import logging

from fastapi import HTTPException, Request

from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.auth.models import User

logger = logging.getLogger(__name__)


async def require_auth(request: Request) -> User:
    """Validate session cookie and return User. Raises 401 if not authenticated."""
    keycloak: KeycloakClient = request.app.state.keycloak

    access_token = request.cookies.get("cce_access")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        return keycloak.decode_token(access_token)
    except Exception:
        # Access token invalid/expired — try refresh
        refresh_token = request.cookies.get("cce_refresh")
        if not refresh_token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            tokens = await keycloak.refresh_token(refresh_token)
            user = keycloak.decode_token(tokens["access_token"])
            request.state.new_access_token = tokens["access_token"]
            request.state.new_refresh_token = tokens.get("refresh_token", refresh_token)
            return user
        except Exception:
            logger.debug("Token refresh failed", exc_info=True)
            raise HTTPException(status_code=401, detail="Session expired")
