"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession


def create_engine(url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine."""
    return create_async_engine(url, echo=False)


async def get_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with SQLModelAsyncSession(engine) as session:
        yield session
