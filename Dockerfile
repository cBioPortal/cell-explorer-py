# ============================================================
# Stage 1: Generate OpenAPI spec from the FastAPI backend
# ============================================================
FROM python:3.12-slim AS openapi

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /backend

# Install workspace dependencies
COPY pyproject.toml .python-version ./
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell-explorer-agent/pyproject.toml packages/cell-explorer-agent/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
COPY packages/zarr-access/pyproject.toml packages/zarr-access/pyproject.toml
COPY packages/zarr-auth-proxy/pyproject.toml packages/zarr-auth-proxy/pyproject.toml
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
ARG FRONTEND_CACHE_BUST=""

WORKDIR /frontend
RUN git clone --depth 1 --branch "$FRONTEND_REF" "$FRONTEND_REPO" .

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy OpenAPI spec from stage 1 and generate api-client types
COPY --from=openapi /backend/openapi.json /tmp/openapi.json
RUN OPENAPI_SPEC=/tmp/openapi.json pnpm --filter @cbioportal-cell-explorer/api-client generate

# Build highperformer
RUN pnpm --filter @cbioportal-cell-explorer/highperformer build

# ============================================================
# Stage 3: Production runtime
# ============================================================
FROM python:3.12-slim AS runtime

ARG GIT_SHA=""
ARG BUILD_DATE=""

LABEL org.opencontainers.image.source="https://github.com/cBioPortal/cell-explorer-py"
LABEL org.opencontainers.image.description="Cell Explorer API and frontend"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="Jason Hwee <hweej@users.noreply.github.com>"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install workspace dependencies
COPY pyproject.toml .python-version ./
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell-explorer-agent/pyproject.toml packages/cell-explorer-agent/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
COPY packages/zarr-access/pyproject.toml packages/zarr-access/pyproject.toml
COPY packages/zarr-auth-proxy/pyproject.toml packages/zarr-auth-proxy/pyproject.toml
RUN uv sync --no-install-workspace

# Copy source and install workspace packages
COPY packages/ packages/
RUN uv sync

# Copy built frontend from stage 2
COPY --from=frontend /frontend/packages/highperformer/dist /app/static

# Configure environment
ENV STATIC_DIR=/app/static
ENV ENVIRONMENT=production
ENV GIT_SHA=${GIT_SHA}
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "cell_explorer_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
