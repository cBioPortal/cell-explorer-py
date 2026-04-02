"""Application info endpoint."""

from importlib.metadata import version

from fastapi import APIRouter
from pydantic import BaseModel

from cell_explorer_api.config import Settings

router = APIRouter()


class InfoResponse(BaseModel):
    version: str
    environment: str
    git_sha: str | None


_settings = Settings()


@router.get("/info")
async def info() -> InfoResponse:
    return InfoResponse(
        version=version("cell-explorer-api"),
        environment=_settings.environment,
        git_sha=_settings.git_sha,
    )
