"""Limits on chat without signing in.

Three counters per question, all in the database so every replica agrees:
per anonymous visitor and per IP (both per clock hour), and per deployment
(per UTC day). The daily one is the ceiling on what signed-out chat can
cost; the hourly ones keep one visitor or one machine from using it all.
Signed-in users never pass through here.
"""

import hashlib
from datetime import datetime, timezone

from sqlmodel import select
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

    Returns the message to show when a limit is already reached (nothing is
    counted then), else counts the question against every limit and returns
    None. The IP is stored only as a SHA-256 hash; without one (no client
    address) the per-IP limit is skipped. Concurrent requests can undercount
    by a question or two, which is fine for a cost guard.
    """
    counters = [("day", "*", _day(now), settings.anon_chat_per_day, DAILY_LIMIT_MESSAGE),
                ("visitor", anon_id, _hour(now), settings.anon_chat_per_visitor_per_hour, HOURLY_LIMIT_MESSAGE)]
    if ip:
        ip_key = hashlib.sha256(ip.encode()).hexdigest()
        counters.append(("ip", ip_key, _hour(now), settings.anon_chat_per_ip_per_hour, HOURLY_LIMIT_MESSAGE))

    rows = []
    for kind, key, window, limit, message in counters:
        row = (await db.exec(select(ChatUsage).where(
            ChatUsage.kind == kind, ChatUsage.key == key, ChatUsage.window_start == window,
        ))).first()
        if row is not None and row.count >= limit:
            return message
        rows.append(row or ChatUsage(kind=kind, key=key, window_start=window, count=0))

    for row in rows:
        row.count += 1
        db.add(row)
    await db.commit()
    return None
