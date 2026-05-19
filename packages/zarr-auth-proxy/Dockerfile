FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install workspace dependencies
COPY pyproject.toml uv.lock .python-version ./
COPY packages/zarr-auth-proxy/pyproject.toml packages/zarr-auth-proxy/pyproject.toml
# Include all other package tomls for workspace resolution — api references
# cell-explorer-agent and zarr-access via `tool.uv.sources = { workspace = true }`,
# so all five workspace members must be present for `uv sync` to resolve.
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
COPY packages/cell-explorer-agent/pyproject.toml packages/cell-explorer-agent/pyproject.toml
COPY packages/zarr-access/pyproject.toml packages/zarr-access/pyproject.toml
RUN uv sync --no-install-workspace

# Copy source and install
COPY packages/zarr-auth-proxy/ packages/zarr-auth-proxy/
# Copy other packages for workspace resolution
COPY packages/api/ packages/api/
COPY packages/cell2zarr/ packages/cell2zarr/
COPY packages/cell-explorer-agent/ packages/cell-explorer-agent/
COPY packages/zarr-access/ packages/zarr-access/
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "zarr_auth_proxy.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
