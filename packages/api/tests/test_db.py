"""Tests for database engine and session factory."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from cell_explorer_api.db import create_engine, get_session


def test_create_engine_returns_async_engine():
    engine = create_engine("sqlite+aiosqlite://")
    assert isinstance(engine, AsyncEngine)


@pytest.mark.asyncio
async def test_get_session_yields_async_session():
    engine = create_engine("sqlite+aiosqlite://")
    async for session in get_session(engine):
        assert isinstance(session, AsyncSession)
