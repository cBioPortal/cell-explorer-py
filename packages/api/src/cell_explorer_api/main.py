"""FastAPI application factory."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cell_explorer_api.config import Settings, validate_static_dir
from cell_explorer_api.routes import router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, reads from environment variables.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(title="Cell Explorer API")
    app.state.settings = settings

    # CORS middleware
    if settings.cors_origin_list:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    # Auth (conditional on Keycloak config)
    if settings.auth_enabled:
        from cell_explorer_api.auth.keycloak import KeycloakClient
        from cell_explorer_api.routes import create_auth_router

        keycloak = KeycloakClient(settings)
        app.state.keycloak = keycloak
        app.include_router(create_auth_router(), prefix="/api")

    # 1. API routes (highest precedence)
    app.include_router(router)

    # 2 & 3. Static serving (if configured)
    if settings.static_dir is not None:
        validated = validate_static_dir(settings.static_dir)

        if validated is not None:
            index_html = validated / "index.html"

            # Mount static assets (e.g. /assets/main.js)
            assets_dir = validated / "assets"
            if assets_dir.is_dir():
                app.mount(
                    "/assets",
                    StaticFiles(directory=str(assets_dir)),
                    name="assets",
                )

            # SPA catch-all: serve index.html for all non-API, non-asset routes
            @app.get("/{path:path}")
            async def spa_catchall(path: str):
                return FileResponse(str(index_html))
        else:
            # STATIC_DIR was set but invalid
            @app.get("/{path:path}")
            async def frontend_not_found(request: Request):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "Frontend not found. Set STATIC_DIR to the path of the built frontend."
                    },
                )

    return app


app = create_app()
