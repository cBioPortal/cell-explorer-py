"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession


def create_engine(url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    For SQLite, enable foreign key enforcement on every connection — SQLite
    has it OFF by default, so ON DELETE CASCADE declarations would silently
    no-op without this pragma.
    """
    engine = create_async_engine(url, echo=False)
    if url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


async def get_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with SQLModelAsyncSession(engine) as session:
        yield session


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a database session from the app's engine."""
    engine: AsyncEngine = request.app.state.db_engine
    async with SQLModelAsyncSession(engine) as session:
        yield session
