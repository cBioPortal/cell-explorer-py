"""Signed-out chat usage limits."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db import create_engine
from cell_explorer_api.db.models import ChatUsage
from cell_explorer_api.services.chat_limits import (
    DAILY_LIMIT_MESSAGE,
    HOURLY_LIMIT_MESSAGE,
    check_and_count,
)

NOON = datetime(2026, 11, 15, 12, 5, tzinfo=timezone.utc)


@pytest_asyncio.fixture()
async def db():
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        yield session


def _settings(**o):
    return Settings(anon_chat_per_visitor_per_hour=o.get("visitor", 20),
                    anon_chat_per_ip_per_hour=o.get("ip", 200),
                    anon_chat_per_day=o.get("day", 10000))


async def _ask(db, n, *, anon="a", ip="1.2.3.4", now=NOON, settings=None):
    out = None
    for _ in range(n):
        out = await check_and_count(db, anon_id=anon, ip=ip, now=now, settings=settings or _settings())
    return out


async def test_visitor_limit(db):
    s = _settings(visitor=3)
    assert await _ask(db, 3, settings=s) is None
    assert await _ask(db, 1, settings=s) == HOURLY_LIMIT_MESSAGE
    assert await _ask(db, 1, now=NOON + timedelta(hours=1), settings=s) is None


async def test_ip_limit_spans_visitors(db):
    s = _settings(ip=3)
    for anon in ("a", "b", "c"):
        assert await _ask(db, 1, anon=anon, settings=s) is None
    assert await _ask(db, 1, anon="d", settings=s) == HOURLY_LIMIT_MESSAGE
    assert await _ask(db, 1, anon="e", ip="5.6.7.8", settings=s) is None


async def test_daily_cap_returns_daily_copy(db):
    s = _settings(day=2)
    assert await _ask(db, 1, anon="a", ip="1.1.1.1", settings=s) is None
    assert await _ask(db, 1, anon="b", ip="2.2.2.2", settings=s) is None
    assert await _ask(db, 1, anon="c", ip="3.3.3.3", settings=s) == DAILY_LIMIT_MESSAGE
    tomorrow = NOON + timedelta(days=1)
    assert await _ask(db, 1, anon="c", ip="3.3.3.3", now=tomorrow, settings=s) is None


async def test_a_refused_question_is_not_counted(db):
    s = _settings(visitor=1)
    await _ask(db, 1, settings=s)
    await _ask(db, 5, settings=s)
    rows = (await db.exec(select(ChatUsage).where(ChatUsage.kind == "visitor"))).all()
    assert [r.count for r in rows] == [1]


async def test_ip_is_hashed_not_stored(db):
    await _ask(db, 1, ip="203.0.113.7")
    keys = [r.key for r in (await db.exec(select(ChatUsage))).all()]
    assert all("203.0.113.7" not in k for k in keys)
    assert any(len(k) == 64 for k in keys)


async def test_missing_ip_skips_ip_limit(db):
    s = _settings(ip=1)
    assert await _ask(db, 3, ip=None, settings=s) is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ANON_CHAT_PER_VISITOR_PER_HOUR", "40")
    monkeypatch.setenv("ANON_CHAT_PER_IP_PER_HOUR", "1000")
    monkeypatch.setenv("ANON_CHAT_PER_DAY", "5000")
    s = Settings()
    assert (s.anon_chat_per_visitor_per_hour, s.anon_chat_per_ip_per_hour, s.anon_chat_per_day) == (40, 1000, 5000)


def test_defaults():
    s = Settings()
    assert (s.anon_chat_per_visitor_per_hour, s.anon_chat_per_ip_per_hour, s.anon_chat_per_day) == (20, 200, 10000)


async def test_concurrent_requests_are_all_counted_without_errors(tmp_path):
    """Parallel requests, each with its own session, as on a busy server."""
    import asyncio as _asyncio

    url = f"sqlite+aiosqlite:///{tmp_path / 'usage.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    s = _settings(visitor=1000, ip=1000, day=10)

    async def one(i):
        async with AsyncSession(engine) as session:
            return await check_and_count(session, anon_id=f"v{i}", ip="9.9.9.9", now=NOON, settings=s)

    results = await _asyncio.gather(*(one(i) for i in range(20)))
    assert results.count(None) == 10
    assert results.count(DAILY_LIMIT_MESSAGE) == 10
    async with AsyncSession(engine) as session:
        day = (await session.exec(select(ChatUsage).where(ChatUsage.kind == "day"))).one()
        assert day.count == 10
    await engine.dispose()
