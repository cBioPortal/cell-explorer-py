"""API route modules."""

from fastapi import APIRouter

from cell_explorer_api.routes.datasets import router as datasets_router
from cell_explorer_api.routes.health import router as health_router
from cell_explorer_api.routes.info import router as info_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(info_router)
router.include_router(datasets_router)


def create_auth_router() -> APIRouter:
    """Import and return the auth router. Separated to allow conditional registration."""
    from cell_explorer_api.routes.auth import router as auth_router
    return auth_router
