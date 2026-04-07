"""Application info endpoint."""

from importlib.metadata import version

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class InfoResponse(BaseModel):
    version: str
    environment: str
    git_sha: str | None
    auth_enabled: bool


@router.get("/info")
async def info(request: Request) -> InfoResponse:
    settings = request.app.state.settings
    return InfoResponse(
        version=version("cell-explorer-api"),
        environment=settings.environment,
        git_sha=settings.git_sha,
        auth_enabled=settings.auth_enabled,
    )
