"""HTTP routes for the chat agent (Plan 2a)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.dependencies import require_auth
from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.db import get_db
from cell_explorer_api.services.chat_session import (
    ChatSessionError,
    make_chat_agent,
)

router = APIRouter(tags=["chat"])

# Note: this router is included by routes/__init__.py whose own router has
# prefix="/api". Routes inside use absolute paths ("/chat/{slug}/context"),
# matching the convention in datasets.py.


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
        # Error mapping is fleshed out in Task 3; for now any setup failure is 500.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
