"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cell_explorer_api.config import Settings, validate_static_dir
from cell_explorer_api.routes import router

logger = logging.getLogger(__name__)


def _configure_file_logging(settings: Settings) -> None:
    """Create data directories and configure rotating file log handler."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    handler = TimedRotatingFileHandler(
        filename=settings.log_dir / settings.log_filename,
        when=settings.log_rotation_when,
        backupCount=settings.log_backup_count,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.addHandler(handler)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, reads from environment variables.
    """
    if settings is None:
        settings = Settings()

    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(settings)

    app = FastAPI(
        title="Cell Explorer API",
        swagger_ui_parameters={"persistAuthorization": True},
    )
    app.state.settings = settings

    # CORS middleware
    if settings.cors_origin_list:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

    # Token refresh middleware — when require_auth refreshes an expired access
    # token, it stores new tokens on request.state. This middleware injects
    # Set-Cookie headers on the response so the browser receives the rotated
    # pair, regardless of which endpoint triggered the refresh. Implemented as
    # pure ASGI middleware (not BaseHTTPMiddleware) so it works correctly with
    # StreamingResponse — the chat /turns endpoint depends on this.
    from cell_explorer_api.auth.middleware import TokenRefreshMiddleware

    app.add_middleware(TokenRefreshMiddleware)

    # Database engine
    from cell_explorer_api.db import create_engine

    app.state.db_engine = create_engine(settings.effective_database_url)

    # Auth — routes always registered (return 501 when disabled); Keycloak client only when configured
    if settings.auth_enabled:
        from cell_explorer_api.auth.keycloak import KeycloakClient

        keycloak = KeycloakClient(settings)
        app.state.keycloak = keycloak

    # Unified lifespan: always dispose db engine; fetch JWKS only when auth enabled
    @asynccontextmanager
    async def lifespan(app):
        if settings.auth_enabled:
            await app.state.keycloak.fetch_jwks()
        yield
        # Flush any queued Langfuse traces before tearing down. The flush
        # is a best-effort call that swallows errors; never raises.
        from cell_explorer_agent.telemetry import langfuse_client
        langfuse_client.flush()
        await app.state.db_engine.dispose()

    app.router.lifespan_context = lifespan

    from cell_explorer_api.routes import create_admin_router, create_auth_router

    app.include_router(create_auth_router(), prefix="/api")
    app.include_router(create_admin_router(), prefix="/api")

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
                if path.startswith("api/"):
                    return JSONResponse(status_code=404, content={"detail": "Not found"})
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
