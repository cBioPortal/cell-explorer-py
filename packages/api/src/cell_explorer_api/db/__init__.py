"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession


def create_engine(url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine."""
    return create_async_engine(url, echo=False)


async def get_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with SQLModelAsyncSession(engine) as session:
        yield session


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a database session from the app's engine."""
    engine: AsyncEngine = request.app.state.db_engine
    async with SQLModelAsyncSession(engine) as session:
        yield session
