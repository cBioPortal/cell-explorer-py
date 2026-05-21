"""DB access for chat threads.

Encapsulates create/list/load/append/delete and title derivation. Raises
ThreadNotFoundError / ThreadAccessDeniedError so the route layer can map
them to HTTP 404 / 403.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db.models import (
    ChatMessageRow,
    ChatThread,
    Dataset,
)


class _UserLike(Protocol):
    sub: str


class ThreadNotFoundError(Exception):
    """Raised when a thread id does not exist in the DB."""


class ThreadAccessDeniedError(Exception):
    """Raised when the user is not the owner of the thread."""


@dataclass(frozen=True)
class ThreadSummary:
    """List-view shape: cheap to compute, suitable for the catalog."""
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


_TITLE_MAX = 50
_WHITESPACE = re.compile(r"\s+")


def derive_title(first_message: str) -> str:
    """Truncate the first user message into a list-view title.

    - Collapses any run of whitespace (incl. newlines) into a single space.
    - Strips leading/trailing whitespace.
    - Truncates to 50 characters, appending `…` if cut.
    """
    collapsed = _WHITESPACE.sub(" ", first_message).strip()
    if len(collapsed) <= _TITLE_MAX:
        return collapsed
    return collapsed[:_TITLE_MAX] + "…"


async def create_thread(
    session: AsyncSession,
    *,
    user: _UserLike,
    dataset: Dataset,
    first_user_message: str,
) -> ChatThread:
    """Create a new thread and add it to the session (does not commit)."""
    thread = ChatThread(
        user_sub=user.sub,
        dataset_id=dataset.id,
        title=derive_title(first_user_message),
    )
    session.add(thread)
    await session.flush()
    return thread


async def list_threads(
    session: AsyncSession, *, user: _UserLike, dataset: Dataset,
) -> list[ThreadSummary]:
    """Return this user's threads on this dataset, newest first."""
    stmt = (
        select(
            ChatThread.id,
            ChatThread.title,
            ChatThread.created_at,
            ChatThread.updated_at,
            func.count(ChatMessageRow.id).label("message_count"),
        )
        .outerjoin(ChatMessageRow, ChatMessageRow.thread_id == ChatThread.id)
        .where(
            ChatThread.user_sub == user.sub,
            ChatThread.dataset_id == dataset.id,
        )
        .group_by(ChatThread.id)
        .order_by(ChatThread.updated_at.desc())
    )
    result = await session.exec(stmt)
    return [
        ThreadSummary(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
            message_count=row.message_count,
        )
        for row in result.all()
    ]


async def load_thread(
    session: AsyncSession, *, user: _UserLike, thread_id: uuid.UUID,
) -> ChatThread:
    """Load a thread by id, enforcing ownership."""
    result = await session.exec(select(ChatThread).where(ChatThread.id == thread_id))
    thread = result.first()
    if thread is None:
        raise ThreadNotFoundError(f"thread {thread_id} not found")
    if thread.user_sub != user.sub:
        raise ThreadAccessDeniedError(f"thread {thread_id} not owned by user")
    return thread


async def load_thread_messages(
    session: AsyncSession, *, thread: ChatThread,
) -> list[ChatMessageRow]:
    """Return the thread's messages ordered by created_at ascending."""
    stmt = (
        select(ChatMessageRow)
        .where(ChatMessageRow.thread_id == thread.id)
        .order_by(ChatMessageRow.created_at.asc())
    )
    result = await session.exec(stmt)
    return list(result.all())


async def append_message(
    session: AsyncSession,
    thread: ChatThread,
    *,
    role: str,
    content: str,
    langfuse_trace_id: str | None = None,
) -> ChatMessageRow:
    """Append a message to the thread and bump its updated_at."""
    msg = ChatMessageRow(
        thread_id=thread.id,
        role=role,
        content=content,
        langfuse_trace_id=langfuse_trace_id,
    )
    session.add(msg)
    thread.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(thread)
    await session.flush()
    return msg


async def delete_thread(
    session: AsyncSession, *, user: _UserLike, thread_id: uuid.UUID,
) -> None:
    """Delete the thread (messages cascade via FK)."""
    thread = await load_thread(session, user=user, thread_id=thread_id)
    await session.delete(thread)
    await session.flush()
