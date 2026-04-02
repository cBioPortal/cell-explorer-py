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

# ============================================================
# Stage 2: Build the frontend from GitHub
# ============================================================
FROM node:22-slim AS frontend

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN corepack enable && corepack prepare pnpm@9.12.1 --activate

ARG FRONTEND_REPO=https://github.com/cBioPortal/cbioportal-cell-explorer.git
ARG FRONTEND_REF=main

WORKDIR /frontend
RUN git clone --depth 1 --branch "$FRONTEND_REF" "$FRONTEND_REPO" .

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy OpenAPI spec from stage 1 and generate api-client types
COPY --from=openapi /backend/openapi.json /tmp/openapi.json
RUN OPENAPI_SPEC=/tmp/openapi.json pnpm --filter @cbioportal-cell-explorer/api-client generate

# Build highperformer
RUN pnpm --filter @cbioportal-cell-explorer/highperformer build
