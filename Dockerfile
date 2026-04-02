# ============================================================
# Stage 1: Generate OpenAPI spec from the FastAPI backend
# ============================================================
FROM python:3.12-slim AS openapi

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /backend

# Install workspace dependencies
COPY pyproject.toml .python-version ./
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
RUN uv sync --no-install-workspace

# Copy source and install workspace packages
COPY packages/ packages/
RUN uv sync

# Generate openapi.json (or use override)
ARG OPENAPI_SPEC_PATH=""
RUN if [ -n "$OPENAPI_SPEC_PATH" ]; then \
        cp "$OPENAPI_SPEC_PATH" /backend/openapi.json; \
    else \
        uv run python -m cell_explorer_api.export_openapi > /backend/openapi.json; \
    fi
