FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Every workspace member's pyproject.toml must be present for uv to resolve the
# lockfile; --package then installs only this server's dependency tree.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell-explorer-agent/pyproject.toml packages/cell-explorer-agent/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
COPY packages/mcp/pyproject.toml packages/mcp/pyproject.toml
COPY packages/zarr-access/pyproject.toml packages/zarr-access/pyproject.toml
COPY packages/zarr-auth-proxy/pyproject.toml packages/zarr-auth-proxy/pyproject.toml
RUN uv sync --frozen --no-dev --package cell-explorer-mcp --no-install-workspace

COPY packages/mcp/ packages/mcp/
COPY packages/zarr-access/ packages/zarr-access/
RUN uv sync --frozen --no-dev --package cell-explorer-mcp

ENV MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_HTTP_PATH=/mcp

EXPOSE 8000

CMD ["/app/.venv/bin/cell-explorer-mcp"]
