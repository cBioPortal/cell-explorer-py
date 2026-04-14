FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install workspace dependencies
COPY pyproject.toml uv.lock .python-version ./
COPY packages/zarr-auth-proxy/pyproject.toml packages/zarr-auth-proxy/pyproject.toml
# Include other package tomls for workspace resolution
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/cell2zarr/pyproject.toml packages/cell2zarr/pyproject.toml
RUN uv sync --no-install-workspace

# Copy source and install
COPY packages/zarr-auth-proxy/ packages/zarr-auth-proxy/
# Copy other packages for workspace resolution
COPY packages/api/ packages/api/
COPY packages/cell2zarr/ packages/cell2zarr/
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "zarr_auth_proxy.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
