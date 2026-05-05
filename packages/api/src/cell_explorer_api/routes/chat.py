"""HTTP routes for the chat agent (Plan 2a)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.db import get_db
from cell_explorer_api.services.chat_session import (
    AccessDeniedError,
    ChatSessionError,
    CredentialMintError,
    DatasetNotFoundError,
    ZarrUnreachableError,
    make_chat_agent,
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


class ContextResponse(BaseModel):
    slug: str
    name: str
    description: str
    n_obs: int
    n_var: int
    obs_columns: list[ObsColumnInfo]
    embedding_keys: list[str]
    available_tools: list[str]


def _http_for_chat_session_error(exc: ChatSessionError) -> HTTPException:
    if isinstance(exc, DatasetNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
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
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> ContextResponse:
    settings: Settings = request.app.state.settings
    try:
        agent = await make_chat_agent(user=user, dataset_slug=slug, db=db, settings=settings)
    except ChatSessionError as exc:
        raise _http_for_chat_session_error(exc) from exc

    ctx = agent.dataset_ctx
    return ContextResponse(
        slug=ctx.slug,
        name=ctx.name,
        description=ctx.description,
        n_obs=ctx.n_obs,
        n_var=ctx.n_var,
        obs_columns=[
            ObsColumnInfo(name=c.name, dtype=c.dtype, cardinality=c.cardinality)
            for c in ctx.obs_columns
        ],
        embedding_keys=list(ctx.embedding_keys),
        available_tools=[t.name for t in agent.catalog.all()],
    )


@router.post("/chat/{slug}/turns")
async def post_chat_turn(
    slug: str,
    body: TurnRequest,
    request: Request,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    # Stub: validation has succeeded by the time we get here. Task 5
    # replaces this with the streaming agent run.
    return {"stub": True}
