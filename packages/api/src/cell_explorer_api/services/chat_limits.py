"""Limits on chat without signing in.

Three counters per question, all in the database so every replica agrees:
per anonymous visitor and per IP (both per clock hour), and per deployment
(per UTC day). The daily one is the ceiling on what signed-out chat can
cost; the hourly ones keep one visitor or one machine from using it all.
Signed-in users never pass through here.
"""

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import ChatUsage

HOURLY_LIMIT_MESSAGE = "You've reached the limit for now — try again in an hour, or sign in to keep going."
DAILY_LIMIT_MESSAGE = "Chat has reached today's limit — try again tomorrow, or sign in to keep going."


def _hour(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)


def _day(now: datetime) -> datetime:
    return _hour(now).replace(hour=0)


async def check_and_count(
    db: AsyncSession,
    *,
    anon_id: str,
    ip: str | None,
    now: datetime,
    settings: Settings,
) -> str | None:
    """Count one signed-out question, or say which limit it would cross.

    Each counter is bumped with a single atomic upsert (INSERT ... ON CONFLICT
    DO UPDATE SET count = count + 1 RETURNING count), all in one transaction,
    so concurrent requests on any replica can neither overwrite each other's
    counts nor collide creating a new window's row. If any counter goes past
    its limit the transaction is rolled back — a refused question is not
    counted — and the message for that limit is returned; otherwise None.
    The IP is stored only as a SHA-256 hash; without one (no client address)
    the per-IP limit is skipped.
    """
    counters = [("day", "*", _day(now), settings.anon_chat_per_day, DAILY_LIMIT_MESSAGE),
                ("visitor", anon_id, _hour(now), settings.anon_chat_per_visitor_per_hour, HOURLY_LIMIT_MESSAGE)]
    if ip:
        ip_key = hashlib.sha256(ip.encode()).hexdigest()
        counters.append(("ip", ip_key, _hour(now), settings.anon_chat_per_ip_per_hour, HOURLY_LIMIT_MESSAGE))

    insert = postgresql_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
    refusal = None
    for kind, key, window, limit, message in counters:
        stmt = (
            insert(ChatUsage)
            .values(id=uuid.uuid4(), kind=kind, key=key, window_start=window, count=1)
            .on_conflict_do_update(
                index_elements=["kind", "key", "window_start"],
                set_={"count": ChatUsage.count + 1},
            )
            .returning(ChatUsage.count)
        )
        count = (await db.exec(stmt)).scalar_one()
        if count > limit and refusal is None:
            refusal = message
    if refusal:
        await db.rollback()
        return refusal
    await db.commit()
    return None


# Size caps on a signed-out turn. The browser sends the whole conversation
# and view state with every question, and all of it goes to the model on
# every round of the agent loop, so without caps one request could cost
# dollars rather than cents and the daily ceiling would bound nothing.
ANON_MAX_MESSAGES = 30
ANON_MAX_MESSAGE_CHARS = 4_000
ANON_MAX_TOTAL_CHARS = 40_000
ANON_MAX_VIEW_STATE_BYTES = 4_096

OVERSIZED_TURN_MESSAGE = (
    "This conversation is too long to continue without signing in — "
    "start a new chat, or sign in to keep going."
)


def oversized_turn(contents: list[str], view_state: dict | None) -> bool:
    """Whether a signed-out turn is over the size caps (checked before counting)."""
    import json

    if len(contents) > ANON_MAX_MESSAGES:
        return True
    if any(len(c) > ANON_MAX_MESSAGE_CHARS for c in contents):
        return True
    if sum(len(c) for c in contents) > ANON_MAX_TOTAL_CHARS:
        return True
    if view_state is not None and len(json.dumps(view_state)) > ANON_MAX_VIEW_STATE_BYTES:
        return True
    return False
