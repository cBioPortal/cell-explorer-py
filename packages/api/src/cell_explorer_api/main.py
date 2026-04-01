"""FastAPI application factory."""

from fastapi import FastAPI

from cell_explorer_api.config import Settings
from cell_explorer_api.routes import router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, reads from environment variables.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(title="Cell Explorer API")
    app.include_router(router)
    return app


app = create_app()
