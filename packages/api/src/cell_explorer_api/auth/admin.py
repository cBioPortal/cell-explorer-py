"""Admin authentication dependency."""

import logging

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def require_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """Validate admin access via API key or Keycloak admin role.

    Raises 501 if admin is not configured, 403 if not authorized.
    """
    settings = request.app.state.settings

    if not settings.admin_enabled:
        raise HTTPException(status_code=501, detail="Admin API is not configured")

    # Try bearer token from Authorization header (API key)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == settings.admin_api_key:
            return

    # Try Keycloak JWT with admin role
    if settings.auth_enabled:
        access_token = request.cookies.get("cce_access")
        if access_token:
            try:
                from cell_explorer_api.auth.keycloak import KeycloakClient

                keycloak: KeycloakClient = request.app.state.keycloak
                user = keycloak.decode_token(access_token)
                if "admin" in user.roles:
                    return
            except Exception:
                pass

    raise HTTPException(status_code=403, detail="Admin access required")
