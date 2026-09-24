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
| `BRAND_DIR` | unset | Directory holding an operator-supplied brand bundle (`brand.json` plus assets). Unset means built-in cBioPortal branding. Mount read-only |
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

## Branding

`BRAND_DIR` (see [Configuration](#configuration) above) points at an operator-supplied
brand bundle: a `brand.json` plus the image assets it references. Setting it re-skins the
tab title, favicon, web app manifest, and the header identity surfaced through
`/api/info` (name, tagline, logo, colors) with the operator's own deployment identity.
It is co-branding, not white-labeling — the "Cell Explorer" name and the cBioPortal
footer attribution are fixed regardless of what `BRAND_DIR` contains; see
[What isn't configurable](#what-isnt-configurable) below.

### Bundle layout

```
/opt/cell-explorer/brand/
├── brand.json
├── logo-light.svg
├── logo-dark.svg
├── favicon.ico
├── favicon.svg
├── icon-192.png
├── icon-512.png
└── apple-touch-icon.png
```

`brand.json` is the only filename that matters — it must be named exactly that. Every
other file in the directory is named whatever the operator likes; `brand.json`'s `logo`
and `favicon` fields say which filename plays which role. Asset files must live directly
in `BRAND_DIR` — no subdirectories.

### `brand.json` schema

Every field is optional and falls back to its own default independently. A one-field
file such as `{"name": "My Institute"}` is valid — everything else keeps the built-in
cBioPortal defaults below.

| Field | Type | Max length | Default |
|---|---|---|---|
| `name` | string | 120 | `"cBioPortal"` |
| `shortName` | string | 40 | the resolved `name` |
| `title` (browser tab title) | string | 160 | `"{name} Cell Explorer"` |
| `tagline` | string | 200 | `"Explore millions of cells in your browser."` |
| `logoHref` | absolute `http(s)` URL | — | none (logo is not a link) |
| `logo.onLight` | filename | — | none |
| `logo.onDark` | filename | — | none |
| `logo.alt` | string | 120 | the resolved `name` |
| `colors.ink` | `#RRGGBB` | — | `"#0d2c48"` |
| `colors.inkDeep` | `#RRGGBB` | — | none (the frontend derives a value from `ink` when unset) |
| `colors.themeColor` | `#RRGGBB` | — | the resolved `ink`, else `"#123a5e"` |
| `favicon.ico` | filename | — | none |
| `favicon.svg` | filename | — | none |
| `favicon.png192` | filename | — | none |
| `favicon.png512` | filename | — | none |
| `favicon.appleTouch` | filename | — | none |

**`colors.ink` must be dark.** It's the background of a light-on-dark identity band —
header text is rendered light on top of it — so a value whose WCAG relative luminance
exceeds `0.35` is rejected outright (the field falls back to the default rather than
shipping unreadable text). `colors.inkDeep` and `colors.themeColor` carry no such
constraint.

**`logo.onLight` and `logo.onDark` are two separate assets, not one logo recolored by
CSS.** Brand marks usually arrive as artwork with a fill already baked in (e.g. a white
knockout mark for a dark header, a full-color mark for a light background), so the
bundle carries both and the shell picks whichever fits the surface it's rendering on.

**Asset filenames** (`logo.onLight`, `logo.onDark`, and every `favicon.*` field) must be
a bare filename: no `/`, no `..` anywhere in the string (not just as a `../` path
segment — `my..logo.svg` is rejected too), and not empty or `.`. The extension must be
one of `.svg`, `.png`, `.ico`, `.jpg`, `.jpeg`, `.webp` (case-insensitive). The file must
also actually exist in `BRAND_DIR` — a validated filename pointing at nothing is dropped
the same as an invalid one.

**`logoHref`**, if set, must be an absolute `http://` or `https://` URL with a host —
anything else (a relative path, a `javascript:` URL, a bare string) is rejected.

### When something is wrong

Every field validates independently and fails soft: an invalid value — wrong type,
malformed hex, oversized text, a filename that fails the traversal/extension checks, an
asset that doesn't exist on disk, an `ink` that isn't dark enough — is dropped with a
warning in the container logs, and that one field falls back to its default. A bad
bundle can make the deployment look wrong; it can never stop the application from
starting.

### Deploying the bundle

Mount the directory **read-only** and point `BRAND_DIR` at the mount:

```yaml
services:
  cell-explorer:
    volumes:
      - /opt/cell-explorer/brand:/brand:ro
    environment:
      BRAND_DIR: /brand
```

The bundle is read once, at container startup — changing `brand.json` or an asset
requires restarting the container, not just replacing the file on disk.

On Kubernetes, the equivalent is a ConfigMap mount:

```yaml
volumes:
  - name: brand
    configMap: { name: brand }
volumeMounts:
  - name: brand
    mountPath: /brand
    readOnly: true
env:
  - name: BRAND_DIR
    value: /brand
```

A ConfigMap base64-encodes binary assets and caps out around 1 MB, so a full favicon
set plus multiple logo variants can approach the limit. Larger bundles need a PVC or a
derived image (`FROM cell-explorer / COPY brand/ /brand`) instead.

### Preparing assets

Brand assets rarely arrive web-ready. A few conversions come up repeatedly:

- **EPS → SVG**: `inkscape in.eps --export-type=svg`. Text in EPS source is typically
  outlined already, so the converted SVG has no font dependency.
- **Favicons need a finished set, not a single SVG.** Safari and Windows still want an
  `.ico`; PWA installs want 192×192 and 512×512 PNGs. Ship all of `favicon.ico`,
  `favicon.svg`, `favicon.png192`, and `favicon.png512` rather than relying on the
  browser to synthesize the rest from one file.
- **Print-derived colors need remapping.** Brand colors supplied as CMYK or spot values
  render differently once naively converted to RGB than the brand's own on-screen
  master. Check hex values against the brand's digital style guide, not the print one,
  before dropping them into `brand.json`.

### What isn't configurable

The cBioPortal footer attribution is fixed and deliberately not a `brand.json` field —
there is no way to move, restyle, or suppress it. This is co-branding: the operator's
identity is primary in the header, and cBioPortal remains visibly present as the
underlying platform. Likewise, "Cell Explorer" itself isn't renamed by any field — the
operator brands the surrounding identity, not the tool.

## License

MIT
