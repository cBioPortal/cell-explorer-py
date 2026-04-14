"""Optional authentication dependency."""

import logging

from fastapi import Request

from cell_explorer_api.auth.models import User

logger = logging.getLogger(__name__)


async def optional_auth(request: Request) -> User | None:
    """Return User if authenticated, None if anonymous. Never raises."""
    settings = request.app.state.settings

    if not settings.auth_enabled:
        return None

    access_token = request.cookies.get("cce_access")
    if not access_token:
        return None

    try:
        from cell_explorer_api.auth.keycloak import KeycloakClient

        keycloak: KeycloakClient = request.app.state.keycloak
        return keycloak.decode_token(access_token)
    except Exception:
        logger.debug("Optional auth: token decode failed, treating as anonymous")
        return None
