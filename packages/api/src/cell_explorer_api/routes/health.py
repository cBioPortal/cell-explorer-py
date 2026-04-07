"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["status"])


@router.get("/health")
async def health():
    return {"status": "ok"}
