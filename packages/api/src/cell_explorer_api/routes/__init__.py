"""API route modules."""

from fastapi import APIRouter

from cell_explorer_api.routes.health import router as health_router
from cell_explorer_api.routes.info import router as info_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(info_router)
