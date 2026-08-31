# cell-explorer-py

Python backend for [cBioPortal Cell Explorer](https://github.com/cBioPortal/cbioportal-cell-explorer). Converts single-cell RNA-seq data (h5ad) to Zarr v3 stores optimized for web-based visualization.

## Packages

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) monorepo:

| Package | Description |
|---------|-------------|
| [`cell2zarr`](packages/cell2zarr/) | h5ad to Zarr conversion pipeline |
| [`cell-explorer-api`](packages/api/) | FastAPI API + static file serving |

## Setup

```bash
uv sync
```

## Preparing your dataset

To convert single-cell RNA-seq data for use with Cell Explorer, see the [cell2zarr documentation](packages/cell2zarr/README.md).

Quick start:

```bash
# Convert h5ad to Zarr
cell2zarr convert input.h5ad output.zarr --two-phase --encoding-config encoding.json

# Add a UMAP embedding to an existing store
cell2zarr add atlas.h5ad atlas.zarr --key obsm/X_umap
```

## Tests

```bash
uv run pytest packages/cell2zarr/tests/ -v
```

## Running the API

```bash
# API only (no static serving)
uv run uvicorn cell_explorer_api.main:app --reload

# With frontend static serving
STATIC_DIR=/path/to/frontend/dist uv run uvicorn cell_explorer_api.main:app --reload

# Export OpenAPI spec
uv run python -m cell_explorer_api.export_openapi > openapi.json
```

## Configuration

All settings are environment variables, read case-insensitively into `Settings`
in `packages/api/src/cell_explorer_api/config.py`, which is the source of truth.
`.env.example` is a copy-paste template for local development;
`DEPLOYMENT.md` covers the operational detail behind several of these.

Most settings are optional, and several features switch themselves on only when
their configuration is present — so an unset value usually means "off" rather
than "broken".

### Serving and identity

| Variable | Default | Notes |
|---|---|---|
| `STATIC_DIR` | unset | Path to the built frontend. Unset means API-only, no static serving |
| `ENVIRONMENT` | `development` | Reported on `/api/info` |
| `GIT_SHA` | auto-detected | Read from git at startup. Set explicitly in containers, where git is unavailable |

### Analytics

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_ANALYTICS_ID` | unset | GA4 measurement id, served to the frontend by `/api/info`. Unset means no analytics script is loaded |

### Data and logging

| Variable | Default | Notes |
|---|---|---|
| `APP_DATA_DIR` | `./data` | Holds the SQLite database and `logs/` |
| `DATABASE_URL` | unset | Defaults to SQLite at `$APP_DATA_DIR/cell_explorer.db` |
| `LOG_LEVEL` | `INFO` | |
| `LOG_ROTATION_INTERVAL` | `daily` | `daily`, `hourly` or `weekly` |
| `LOG_BACKUP_COUNT` | `30` | Rotated files retained |
| `LOG_FILENAME` | `cell-explorer.log` | Written under `$APP_DATA_DIR/logs/` |

### Authentication

**Auth is enabled only when an issuer, a client id and a client secret all
resolve.** Any one missing leaves auth off and every dataset public — there is
no partial state and no error, so confirm `/api/info` reports
`auth_enabled: true` after configuring it.

`AUTH_PROVIDER` selects how the issuer is derived and which claims carry roles.

| Variable | Default | Notes |
|---|---|---|
| `AUTH_PROVIDER` | `keycloak` | `keycloak`, `entra` or `oidc` |
| `OIDC_ISSUER` | unset | Required unless `AUTH_PROVIDER=keycloak`, which derives it from `KEYCLOAK_URL` + `KEYCLOAK_REALM` |
| `OIDC_CLIENT_ID` | unset | Falls back to `KEYCLOAK_CLIENT_ID` |
| `OIDC_CLIENT_SECRET` | unset | Falls back to `KEYCLOAK_CLIENT_SECRET` |
| `OIDC_SCOPES` | `openid profile email` | `entra` appends `offline_access` automatically |
| `OIDC_AUDIENCE` | unset | Defaults to the resolved client id |
| `OIDC_ROLES_CLAIMS` | unset | Comma-separated dotted claim paths merged into the user's roles. Defaults per provider: Keycloak uses `realm_access.roles` and `resource_access.<client>.roles`, Entra uses `roles` |

The `KEYCLOAK_*` variables are the zero-config path for `AUTH_PROVIDER=keycloak`,
and remain supported aliases for the generic names above.

| Variable | Default | Notes |
|---|---|---|
| `KEYCLOAK_URL` | unset | Base URL; combined with the realm to derive the issuer |
| `KEYCLOAK_REALM` | unset | |
| `KEYCLOAK_CLIENT_ID` | unset | |
| `KEYCLOAK_CLIENT_SECRET` | unset | |
| `KEYCLOAK_IDP_HINT` | unset | Skips the provider chooser. Ignored unless `AUTH_PROVIDER=keycloak` |

### Sessions and CORS

| Variable | Default | Notes |
|---|---|---|
| `ACCESS_COOKIE_MAX_AGE` | `300` | Seconds. 5 minutes |
| `REFRESH_COOKIE_MAX_AGE` | `86400` | Seconds. 24 hours. **Keep this at or below the realm's `ssoSessionMaxLifespan`**, or refresh fails early and users see "Session expired" mid-session |
| `CORS_ORIGINS` | unset | Comma-separated origins. Unset means no CORS middleware is installed |

### Admin, chat and CLI

| Variable | Default | Notes |
|---|---|---|
| `ADMIN_API_KEY` | unset | Enables `/api/admin/*`. Unset means the admin endpoints reject every request |
| `ANTHROPIC_API_KEY` | unset | Unset means chat is disabled and `/api/info` reports `chat_enabled: false` |
| `CHAT_REQUIRED_ROLE` | unset | Role required for chat. Unset means any authenticated user, still subject to each dataset's own `chat_enabled` |
| `CLI_STATE_SECRET` | unset | Signs the CLI login callback state |

## License

MIT
