"""HTTP routes for the chat agent (Plan 2a)."""

import asyncio
import uuid as _uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_agent.events import Error, ThreadOpen
from cell_explorer_agent.messages import AssistantMessage, UserMessage
from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.config import Settings
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import (
    ChatFeedback,
    ChatMessageRow,
    ChatThread,
    Dataset,
    _utcnow,
)
from cell_explorer_api.services.access import compute_chat_permission
from cell_explorer_api.services.chat_session import (
    AccessDeniedError,
    ChatDisabledError,
    ChatSessionError,
    CredentialMintError,
    DatasetNotFoundError,
    ZarrUnreachableError,
    assert_chat_access,
    make_chat_agent,
)
from cell_explorer_api.services.threads import (
    ThreadAccessDeniedError,
    ThreadNotFoundError,
    append_message,
    create_thread,
    delete_thread,
    list_threads,
    load_thread,
    load_thread_messages,
)

router = APIRouter(tags=["chat"])

# Note: this router is included by routes/__init__.py whose own router has
# prefix="/api". Routes inside use absolute paths ("/chat/{slug}/context"),
# matching the convention in datasets.py.


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class TurnRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    view_state: dict[str, Any] | None = None
    thread_id: _uuid.UUID | None = None

    @field_validator("messages")
    @classmethod
    def _validate_alternating_ending_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != "user":
            raise ValueError("last message must have role='user'")
        for i, m in enumerate(v):
            expected = "user" if i % 2 == 0 else "assistant"
            if m.role != expected:
                raise ValueError(
                    f"messages[{i}].role must be {expected!r} (alternating from user)"
                )
        return v


class ObsColumnInfo(BaseModel):
    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None = None
    values: list[str] | None = None


class ChatPermissionResponse(BaseModel):
    can_chat: bool
    reason: str | None = None


class ContextResponse(BaseModel):
    slug: str
    name: str
    description: str
    n_obs: int
    n_var: int
    obs_columns: list[ObsColumnInfo]
    embedding_keys: list[str]
    available_tools: list[str]
    permission: ChatPermissionResponse


class ThreadSummaryResponse(BaseModel):
    id: _uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummaryResponse]


class ThreadMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ThreadDetailResponse(BaseModel):
    id: _uuid.UUID
    title: str
    messages: list[ThreadMessageResponse]


def _to_agent_messages(messages: list[ChatMessage]):
    out = []
    for m in messages:
        if m.role == "user":
            out.append(UserMessage(content=m.content))
        else:
            out.append(AssistantMessage(text=m.content))
    return out


async def _ndjson_event_stream(
    agent,
    messages,
    view_state: dict[str, Any] | None,
    *,
    engine,  # AsyncEngine — used to open a fresh session for stream-time writes
    thread_id: _uuid.UUID,
    thread_title: str,
) -> AsyncIterator[bytes]:
    """Serialize agent events as NDJSON. Emits a leading thread_open event,
    accumulates assistant text deltas, and persists the assistant's final
    body when the stream finishes successfully.
    """
    # 1) Leading thread_open.
    thread_open = ThreadOpen(thread_id=str(thread_id), title=thread_title)
    yield (thread_open.model_dump_json() + "\n").encode()

    assistant_buffer: list[str] = []
    try:
        async for event in agent.run(messages=messages, view_state=view_state):
            if event.type == "text_delta":
                assistant_buffer.append(event.text)
            yield (event.model_dump_json() + "\n").encode()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — leak becomes an event
        err = Error(message=f"chat failed: {exc}", retryable=False)
        yield (err.model_dump_json() + "\n").encode()
        return

    # After the agent's stream is fully consumed, persist the assistant message.
    # This runs after the client has already received the done event.
    if assistant_buffer:
        async with AsyncSession(engine) as stream_db:
            thread = (await stream_db.exec(
                select(ChatThread).where(ChatThread.id == thread_id)
            )).first()
            if thread is not None:
                await append_message(
                    stream_db, thread, role="assistant",
                    content="".join(assistant_buffer),
                )
                await stream_db.commit()


def _http_for_chat_session_error(exc: ChatSessionError) -> HTTPException:
    if isinstance(exc, DatasetNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ChatDisabledError):
        return HTTPException(
            status_code=404,
            detail="Chat is not available for this dataset",
        )
    if isinstance(exc, AccessDeniedError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, CredentialMintError):
        return HTTPException(
            status_code=502,
            detail=f"could not mint datasource credentials: {exc}",
        )
    if isinstance(exc, ZarrUnreachableError):
        return HTTPException(status_code=502, detail=f"could not reach zarr store: {exc}")
    return HTTPException(status_code=500, detail="internal error")


@router.get("/chat/{slug}/context", response_model=ContextResponse)
async def get_chat_context(
    slug: str,
    request: Request,
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> ContextResponse:
    settings: Settings = request.app.state.settings
    if not settings.chat_enabled:
        raise HTTPException(status_code=404, detail="Chat is not available")
    try:
        agent = await make_chat_agent(
            user=user, dataset_slug=slug, db=db, settings=settings,
        )
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    permission = compute_chat_permission(
        user, required_role=settings.chat_required_role,
    )
    ctx = agent.dataset_ctx
    return ContextResponse(
        slug=ctx.slug,
        name=ctx.name,
        description=ctx.description,
        n_obs=ctx.n_obs,
        n_var=ctx.n_var,
        obs_columns=[
            ObsColumnInfo(
                name=c.name,
                dtype=c.dtype,
                cardinality=c.cardinality,
                values=c.values,
            )
            for c in ctx.obs_columns
        ],
        embedding_keys=list(ctx.embedding_keys),
        available_tools=[t.name for t in agent.catalog.all()],
        permission=ChatPermissionResponse(
            can_chat=permission.can_chat,
            reason=permission.reason,
        ),
    )


@router.post("/chat/{slug}/turns")
async def post_chat_turn(
    slug: str,
    body: TurnRequest,
    request: Request,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    settings: Settings = request.app.state.settings
    if not settings.chat_enabled:
        raise HTTPException(status_code=404, detail="Chat is not available")
    try:
        agent = await make_chat_agent(user=user, dataset_slug=slug, db=db, settings=settings)
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    # Defense-in-depth: enforce the global chat role even though the
    # frontend normally disables the input when can_chat=False.
    permission = compute_chat_permission(
        user, required_role=settings.chat_required_role,
    )
    if not permission.can_chat:
        raise HTTPException(
            status_code=403,
            detail=f"missing chat role: {settings.chat_required_role}",
        )

    # The dataset row was already looked up inside make_chat_agent; re-fetch
    # here for the FK on chat_threads (cheap, single-row).
    dataset = (await db.exec(select(Dataset).where(Dataset.slug == slug))).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"dataset {slug!r} not found")

    # Last user message is the one we persist (history is full conversation;
    # the actual NEW user input is the last entry).
    last_user_message = body.messages[-1].content

    # Resolve thread: create new, or load existing (with owner check).
    if body.thread_id is None:
        thread = await create_thread(
            db, user=user, dataset=dataset, first_user_message=last_user_message,
        )
    else:
        try:
            thread = await load_thread(db, user=user, thread_id=body.thread_id)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ThreadAccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    # Persist the user message BEFORE the LLM call so a mid-stream crash
    # doesn't lose it.
    await append_message(db, thread, role="user", content=last_user_message)

    # Capture plain values now, before commit() expires the ORM attributes.
    # The async generator runs after this handler returns, outside the
    # request's db session scope, so it uses its own fresh session.
    thread_id_uuid = thread.id
    thread_title_str = thread.title
    engine = request.app.state.db_engine

    await db.commit()

    return StreamingResponse(
        _ndjson_event_stream(
            agent,
            _to_agent_messages(body.messages),
            body.view_state,
            engine=engine,
            thread_id=thread_id_uuid,
            thread_title=thread_title_str,
        ),
        media_type="application/x-ndjson",
    )


@router.get("/chat/{slug}/threads", response_model=ThreadListResponse)
async def get_chat_threads(
    slug: str,
    request: Request,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> ThreadListResponse:
    settings: Settings = request.app.state.settings
    if not settings.chat_enabled:
        raise HTTPException(status_code=404, detail="Chat is not available")

    # Lightweight gate: dataset lookup + access check + chat_enabled flag,
    # without opening the zarr store. /threads doesn't need a live LLM
    # session — it just lists DB rows the user already owns.
    try:
        dataset = await assert_chat_access(user=user, dataset_slug=slug, db=db)
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    summaries = await list_threads(db, user=user, dataset=dataset)
    return ThreadListResponse(
        threads=[
            ThreadSummaryResponse(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=s.message_count,
            )
            for s in summaries
        ],
    )


@router.get("/chat/{slug}/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_chat_thread_detail(
    slug: str,
    thread_id: _uuid.UUID,
    request: Request,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> ThreadDetailResponse:
    settings: Settings = request.app.state.settings
    if not settings.chat_enabled:
        raise HTTPException(status_code=404, detail="Chat is not available")
    try:
        await assert_chat_access(user=user, dataset_slug=slug, db=db)
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    try:
        thread = await load_thread(db, user=user, thread_id=thread_id)
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ThreadAccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    messages = await load_thread_messages(db, thread=thread)
    return ThreadDetailResponse(
        id=thread.id,
        title=thread.title,
        messages=[
            ThreadMessageResponse(
                role=m.role,  # type: ignore[arg-type] — DB stores str; narrow at route
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete("/chat/{slug}/threads/{thread_id}", status_code=204)
async def delete_chat_thread(
    slug: str,
    thread_id: _uuid.UUID,
    request: Request,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.chat_enabled:
        raise HTTPException(status_code=404, detail="Chat is not available")
    try:
        await assert_chat_access(user=user, dataset_slug=slug, db=db)
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    try:
        await delete_thread(db, user=user, thread_id=thread_id)
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ThreadAccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await db.commit()


# ---------- message feedback ----------


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None
    updated_at: datetime


@router.put(
    "/chat/{slug}/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
)
async def put_message_feedback(
    slug: str,
    message_id: _uuid.UUID,
    body: FeedbackRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    # Resolve message; verify ownership via thread.user_sub. We deliberately
    # do NOT 404 on the dataset/slug here — chat_enabled may have been turned
    # off after the user received the message, but they still own it.
    msg = (
        await db.exec(select(ChatMessageRow).where(ChatMessageRow.id == message_id))
    ).first()
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")
    thread = (
        await db.exec(select(ChatThread).where(ChatThread.id == msg.thread_id))
    ).first()
    if thread is None or thread.user_sub != user.sub:
        # Hide existence from non-owners.
        raise HTTPException(status_code=404, detail="Message not found")
    if msg.role != "assistant":
        raise HTTPException(
            status_code=422, detail="Only assistant messages can be rated"
        )

    # Upsert by (message_id, user_sub).
    existing = (
        await db.exec(
            select(ChatFeedback).where(
                ChatFeedback.message_id == message_id,
                ChatFeedback.user_sub == user.sub,
            )
        )
    ).first()
    now = _utcnow()
    if existing is None:
        fb = ChatFeedback(
            message_id=message_id,
            user_sub=user.sub,
            rating=body.rating,
            comment=body.comment,
        )
        db.add(fb)
    else:
        existing.rating = body.rating
        existing.comment = body.comment
        existing.updated_at = now
        fb = existing
    await db.commit()
    await db.refresh(fb)
    return FeedbackResponse(
        rating=fb.rating,  # type: ignore[arg-type]
        comment=fb.comment,
        updated_at=fb.updated_at,
    )


@router.delete(
    "/chat/{slug}/messages/{message_id}/feedback",
    status_code=204,
)
async def delete_message_feedback(
    slug: str,
    message_id: _uuid.UUID,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> Response:
    # Resolve message + thread ownership before deleting feedback.
    msg_stmt = select(ChatMessageRow).where(ChatMessageRow.id == message_id)
    msg = (await db.exec(msg_stmt)).first()
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")
    thread_stmt = select(ChatThread).where(ChatThread.id == msg.thread_id)
    thread = (await db.exec(thread_stmt)).first()
    if thread is None or thread.user_sub != user.sub:
        raise HTTPException(status_code=404, detail="Message not found")

    fb_stmt = select(ChatFeedback).where(
        ChatFeedback.message_id == message_id,
        ChatFeedback.user_sub == user.sub,
    )
    fb = (await db.exec(fb_stmt)).first()
    if fb is None:
        raise HTTPException(status_code=404, detail="No feedback to clear")
    await db.delete(fb)
    await db.commit()
    return Response(status_code=204)
