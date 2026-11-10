"""Chat without signing in, on datasets that allow it.

A signed-out visitor on such a dataset gets a random anonymous ID, kept in
the `cce_anon` cookie for 7 days, and is represented to the chat routes as
a `User` whose `sub` is `anon:<id>`. Threads, messages and feedback are
keyed on `sub`, so everything downstream works unchanged and each anonymous
visitor sees only their own conversations. Signing in moves them into the
account (see the auth callback).
"""

import uuid

from fastapi import Depends, HTTPException, Request, Response
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset
from cell_explorer_api.db import get_db

ANON_COOKIE = "cce_anon"
ANON_PREFIX = "anon:"
ANON_COOKIE_MAX_AGE = 7 * 86400


def is_anonymous(user: User) -> bool:
    return user.sub.startswith(ANON_PREFIX)


def anonymous_chat_allowed(dataset: Dataset | None, settings: Settings) -> bool:
    """Signed-out chat needs the dataset's switch, chat, a public dataset, and no chat role.

    A chat role gates signed-in users; a signed-out visitor never gets more
    than a signed-in user without that role, so a role turns this off.
    """
    return bool(
        dataset is not None
        and dataset.anonymous_chat
        and dataset.chat_enabled
        and dataset.is_public
        and settings.chat_enabled
        and settings.chat_required_role is None
    )


def anon_id_from_cookie(request: Request) -> str | None:
    """The visitor's anonymous ID, or None when the cookie is missing or not a UUID."""
    raw = request.cookies.get(ANON_COOKIE)
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


def set_anon_cookie(request: Request, response: Response, anon_id: str) -> None:
    """Set the anonymous chat cookie: 7 days, HttpOnly, sent with API calls only."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        ANON_COOKIE,
        anon_id,
        max_age=ANON_COOKIE_MAX_AGE,
        httponly=True,
        secure=proto == "https",
        samesite="lax",
        path="/api",
    )


async def chat_user(slug: str, request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """The signed-in user, or an anonymous visitor where the dataset allows it.

    A session is tried first: an expired-but-refreshable one refreshes as
    usual rather than quietly becoming anonymous. Only when there is no
    usable session does an allowed dataset get an anonymous identity; a new
    one is left on `request.state` for the middleware to set as a cookie
    (it must also reach the streaming /turns response).
    """
    if request.cookies.get("cce_access") or request.cookies.get("cce_refresh"):
        try:
            return await require_auth(request)
        except HTTPException as exc:
            if exc.status_code not in (401, 501):
                raise

    settings: Settings = request.app.state.settings
    dataset = (await db.exec(select(Dataset).where(Dataset.slug == slug))).first()
    if not anonymous_chat_allowed(dataset, settings):
        raise HTTPException(status_code=401, detail="Not authenticated")

    anon_id = anon_id_from_cookie(request)
    if anon_id is None:
        anon_id = str(uuid.uuid4())
        request.state.anon_cookie_id = anon_id
    return User(sub=f"{ANON_PREFIX}{anon_id}", roles=[])
