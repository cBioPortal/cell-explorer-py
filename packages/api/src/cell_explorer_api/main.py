"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from cell_explorer_api.branding import load_bundle
from cell_explorer_api.config import Settings, validate_static_dir
from cell_explorer_api.routes import router
from cell_explorer_api.shell import render_index_html, render_webmanifest

logger = logging.getLogger(__name__)


def _resolve_static_file(static_dir: Path, path: str) -> Path | None:
    """Map a request path to a real file under static_dir, or None.

    Returns None for empty paths, directories, missing files, and anything
    that resolves outside static_dir (traversal).
    """
    if not path:
        return None
    try:
        root = static_dir.resolve()
        candidate = (root / path).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None


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
    brand_bundle = load_bundle(settings.brand_dir)
    app.state.brand = brand_bundle.brand

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

    # Auth — routes always registered (return 501 when disabled); OIDC client only when configured
    if settings.auth_enabled:
        from cell_explorer_api.auth.oidc import OidcClient

        oidc = OidcClient(settings)
        app.state.oidc = oidc

    # Unified lifespan: always dispose db engine; fetch JWKS only when auth enabled
    @asynccontextmanager
    async def lifespan(app):
        if settings.auth_enabled:
            await app.state.oidc.discover()
            await app.state.oidc.fetch_jwks()
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

    # 1b. Operator-supplied brand assets. Registered before the SPA catch-all so
    # /brand/* resolves to the bundle rather than falling through to index.html.
    #
    # Only the filenames brand.json actually resolved onto are served. BRAND_DIR
    # is as often as not an operator's working directory, and a blanket mount
    # would publish whatever else is sitting in it — a stray .env, a key, a
    # draft asset — none of which the extension allowlist constrains, because
    # that allowlist governs what brand.json may *reference*, not what the
    # directory *contains*. The bundle decides; BRAND_DIR is never re-stat'd
    # here, so the mount decision cannot diverge from the load decision (and
    # cannot raise EACCES where the load warned).
    if brand_bundle.directory is not None:
        brand_root = brand_bundle.directory
        brand_assets = brand_bundle.asset_filenames

        @app.get("/brand/{filename:path}", include_in_schema=False)
        async def brand_asset(filename: str):
            if filename not in brand_assets:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            # Resolved even though the name is a validated bare filename: it may
            # still be a symlink pointing out of the bundle.
            file = _resolve_static_file(brand_root, filename)
            if file is None:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            return FileResponse(str(file))

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

            # The shell carries brand values the browser needs before any
            # JavaScript runs, so it is templated once here rather than per
            # request — the brand cannot change without a restart.
            # Reading the shell is new work at startup — the old FileResponse
            # never decoded it. An unreadable or undecodable file degrades to
            # exactly the previous behavior rather than stopping the app.
            brand = app.state.brand
            index_path = index_html.resolve()
            try:
                index_body = render_index_html(
                    index_html.read_text(encoding="utf-8"), brand
                )
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not read %s (%s); serving it unbranded and uncached-as-before",
                    index_html,
                    exc,
                )
                index_body = None

            manifest_path = validated / "site.webmanifest"
            manifest_body = None
            if manifest_path.is_file():
                try:
                    manifest_body = render_webmanifest(
                        manifest_path.read_text(encoding="utf-8"), brand
                    )
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Could not read %s (%s); serving it as a static file",
                        manifest_path,
                        exc,
                    )

            # SPA catch-all: serve the templated shell and webmanifest, real
            # files from the static root (Vite copies public/ there — favicons),
            # else index.html.
            @app.get("/{path:path}")
            async def spa_catchall(path: str):
                if path.startswith("api/"):
                    return JSONResponse(status_code=404, content={"detail": "Not found"})
                if path == "site.webmanifest" and manifest_body is not None:
                    return Response(
                        manifest_body,
                        media_type="application/manifest+json",
                        headers={"Cache-Control": "no-cache"},
                    )
                file = _resolve_static_file(validated, path)
                if file is not None and file != index_path:
                    return FileResponse(str(file))
                if index_body is None:
                    return FileResponse(str(index_html))
                return HTMLResponse(index_body, headers={"Cache-Control": "no-cache"})
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
