"""Bearer API tokens on the dataset routes (MCP callers)."""

from fastapi import Depends, Request
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.db import get_db
from cell_explorer_api.services import oauth_store


async def user_or_mcp_caller(
    request: Request,
    cookie_user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Website cookie user if any, else the user behind a valid API token.

    Only ``kind="api"`` tokens are accepted; anything else is anonymous.
    Never sets cookies and never refreshes anything.
    """
    if cookie_user is not None:
        return cookie_user
    if not request.app.state.settings.oauth_enabled:
        return None
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    value = value.strip()
    if scheme.lower() != "bearer" or not value:
        return None
    found = await oauth_store.find_token(db, value, kind="api")
    if found is None:
        return None
    _, grant = found
    return User(
        sub=grant.user_sub,
        email=grant.user_email,
        name=grant.user_name,
        roles=list(grant.roles),
        via_mcp=True,
    )
