"""Zarr auth proxy — JWT-protected file server."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from zarr_auth_proxy.auth import is_path_authorized, validate_token
from zarr_auth_proxy.config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    # Read public key at startup
    public_key = settings.public_key_file.read_bytes()

    app = FastAPI(title="Zarr Auth Proxy")
    app.state.settings = settings
    app.state.public_key = public_key

    # CORS
    if settings.cors_origin_list:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/{path:path}")
    async def serve_file(path: str, request: Request):
        # Extract bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        token = auth_header[7:]

        # Validate JWT
        claims = validate_token(token, app.state.public_key)
        if claims is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})

        # Check path authorization
        token_path = claims.get("path", "")
        if not is_path_authorized(path, token_path):
            return JSONResponse(status_code=401, content={"detail": "Path not authorized"})

        # Resolve file path safely
        data_dir = settings.data_dir.resolve()
        file_path = (data_dir / path).resolve()

        # Prevent directory traversal
        try:
            file_path.relative_to(data_dir)
        except ValueError:
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        if not file_path.is_file():
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        return FileResponse(file_path)

    return app
