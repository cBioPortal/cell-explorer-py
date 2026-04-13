# Keycloak Authentication Setup

This guide covers how to configure Keycloak as the OIDC provider for the `cell-explorer-api`
FastAPI backend. It is intended for developers running local environments and for ops teams
deploying to production.

Authentication is optional. If any of the four required Keycloak environment variables are absent,
auth endpoints respond with `501 Not Implemented` and the API operates without access control.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Creating a Keycloak Realm](#2-creating-a-keycloak-realm)
3. [Creating a Confidential OIDC Client](#3-creating-a-confidential-oidc-client)
4. [Adding an Audience Mapper](#4-adding-an-audience-mapper)
5. [Creating a Test User](#5-creating-a-test-user)
6. [Environment Variables](#6-environment-variables)
7. [Running with Docker Compose](#7-running-with-docker-compose)
8. [Running without Docker](#8-running-without-docker)
9. [Verifying Auth Works](#9-verifying-auth-works)
10. [Auth Endpoints Reference](#10-auth-endpoints-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- A running Keycloak instance with admin access. The cBioPortal production instance is at
  `https://keycloak.cbioportal.org/auth`, but this guide applies to any Keycloak server.
- Permissions to create realms and clients in the Keycloak Admin Console.
- The `cell-explorer-api` backend source (this repository).

---

## 2. Creating a Keycloak Realm

Creating a dedicated realm isolates cell-explorer's users and clients from other applications
sharing the same Keycloak server.

1. Log in to the Keycloak Admin Console.
2. In the top-left realm dropdown, click **Create Realm**.
3. Set **Realm name** to `cell-explorer` (or any name that fits your deployment convention).
4. Click **Create**.

All subsequent steps in this guide take place inside this realm.

---

## 3. Creating a Confidential OIDC Client

The backend performs a server-side authorization code exchange, so it needs a **confidential**
client (one with a client secret). Public clients and implicit flow must not be used.

1. In your realm, go to **Clients** > **Create client**.
2. Set **Client ID** to `cell-explorer-app` (you can choose any ID; just keep it consistent with
   your env vars).
3. Set **Client Protocol** to `openid-connect`.
4. Click **Next**.
5. On the **Capability config** page:
   - **Client authentication**: ON
   - **Standard flow**: enabled
   - **Direct access grants**: disabled
   - **Implicit flow**: disabled
6. Click **Next**.
7. On the **Login settings** page, set redirect URIs and web origins:

   | Setting | Local dev | Production |
   |---|---|---|
   | **Valid Redirect URIs** | `http://localhost:8001/*` | `https://your-domain.example.com/*` |
   | **Web Origins** | `http://localhost:8001` | `https://your-domain.example.com` |

8. Click **Save**.
9. Navigate to the **Credentials** tab for the newly created client. Copy the **Client secret** —
   you will need it for `KEYCLOAK_CLIENT_SECRET`.

---

## 4. Adding an Audience Mapper

**This step is required.** It is the most commonly missed configuration and the source of most
authentication failures.

By default, Keycloak sets the `aud` (audience) claim in access tokens to `account`. The backend
validates that the token's `aud` matches the configured client ID. Without this mapper, every
token will be rejected with an "Audience doesn't match" error.

1. In the client you just created, go to the **Client scopes** tab.
2. Click the link for the dedicated scope (named `cell-explorer-app-dedicated`).
3. Click **Add mapper** > **By configuration**.
4. Select **Audience**.
5. Configure the mapper:

   | Field | Value |
   |---|---|
   | **Name** | `cell-explorer-app-audience` |
   | **Included Client Audience** | `cell-explorer-app` (must match your Client ID exactly) |
   | **Add to ID token** | OFF |
   | **Add to access token** | ON |

6. Click **Save**.

After this mapper is in place, access tokens issued for this client will include
`"aud": ["cell-explorer-app", "account"]`, which the backend will accept.

---

## 5. Creating a Test User

1. In your realm, go to **Users** > **Add user**.
2. Fill in a **Username** and click **Create**.
3. Go to the **Credentials** tab for the new user.
4. Click **Set password**, enter a password, and set **Temporary** to **OFF** (so you are not
   prompted to change it on first login).
5. Click **Save**.

---

## 6. Environment Variables

The backend reads its Keycloak configuration from these environment variables. All four Keycloak
variables must be set for auth to be enabled.

| Variable | Required for auth | Description | Example |
|---|---|---|---|
| `KEYCLOAK_URL` | Yes | Base URL of the Keycloak server | `https://keycloak.example.com/auth` |
| `KEYCLOAK_REALM` | Yes | Name of the realm | `cell-explorer` |
| `KEYCLOAK_CLIENT_ID` | Yes | OIDC client ID | `cell-explorer-app` |
| `KEYCLOAK_CLIENT_SECRET` | Yes | Client secret from the Credentials tab | `9b9b6ac1-...` |
| `CORS_ORIGINS` | No | Comma-separated list of allowed origins | `http://localhost:8001` |
| `STATIC_DIR` | No | Filesystem path to the frontend dist | `/app/static` |
| `ENVIRONMENT` | No | `development` or `production` | `development` |

If any of the four Keycloak variables are missing or empty, the auth subsystem is disabled and all
`/api/auth/*` endpoints return `501 Not Implemented`.

---

## 7. Running with Docker Compose

The `docker-compose.yml` at the repository root reads environment variables from an `.env` file
via `env_file: .env`. This file should never be committed.

**Create `.env` in the repository root:**

```dotenv
KEYCLOAK_URL=https://keycloak.example.com/auth
KEYCLOAK_REALM=cell-explorer
KEYCLOAK_CLIENT_ID=cell-explorer-app
KEYCLOAK_CLIENT_SECRET=9b9b6ac1-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Optional
STATIC_DIR=/app/static
CORS_ORIGINS=http://localhost:8001
ENVIRONMENT=development
```

**Start the API:**

```bash
docker compose up --build
```

The API will be available at `http://localhost:8001`. The `.env` file is already covered by
`.gitignore` — verify this before committing anything in the project root.

---

## 8. Running without Docker

Set the variables inline when launching uvicorn directly:

```bash
KEYCLOAK_URL=https://keycloak.example.com/auth \
KEYCLOAK_REALM=cell-explorer \
KEYCLOAK_CLIENT_ID=cell-explorer-app \
KEYCLOAK_CLIENT_SECRET=9b9b6ac1-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
CORS_ORIGINS=http://localhost:8001 \
ENVIRONMENT=development \
uv run uvicorn cell_explorer_api.main:app --host 0.0.0.0 --port 8001 --reload
```

Or export them in your shell session before running uvicorn:

```bash
export KEYCLOAK_URL=https://keycloak.example.com/auth
export KEYCLOAK_REALM=cell-explorer
export KEYCLOAK_CLIENT_ID=cell-explorer-app
export KEYCLOAK_CLIENT_SECRET=9b9b6ac1-xxxx-xxxx-xxxx-xxxxxxxxxxxx
uv run uvicorn cell_explorer_api.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## 9. Verifying Auth Works

With the API running, run these checks in order:

**1. Confirm auth is enabled:**

```bash
curl http://localhost:8001/api/info
```

Expected response includes `"auth_enabled": true`. If it shows `false`, one or more Keycloak env
vars are missing — see [Troubleshooting](#11-troubleshooting).

**2. Trigger a login redirect:**

Open `http://localhost:8001/api/auth/login` in a browser. You should be redirected to the
Keycloak login page for your realm.

**3. Check the authenticated user:**

After completing login, navigate to:

```
http://localhost:8001/api/auth/me
```

You should receive a JSON object with the authenticated user's claims (sub, email, name, etc.).

---

## 10. Auth Endpoints Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/auth/login` | Redirects the browser to Keycloak to begin the login flow |
| `GET` | `/api/auth/callback` | OAuth2 callback URL; Keycloak redirects here after login (handled automatically) |
| `GET` | `/api/auth/me` | Returns the current authenticated user's claims; requires a valid session |
| `POST` | `/api/auth/logout` | Clears session cookies and logs the user out |
| `POST` | `/api/auth/token-exchange` | Accepts an external JWT and exchanges it for session cookies |

All endpoints return `501 Not Implemented` when auth is not configured.

---

## 11. Troubleshooting

**"Audience doesn't match" error when logging in**

The audience mapper is missing or misconfigured. Follow [Section 4](#4-adding-an-audience-mapper)
carefully. Confirm that **Included Client Audience** exactly matches the `KEYCLOAK_CLIENT_ID`
value.

**Auth endpoints return `501 Not Implemented`**

Not all four Keycloak environment variables are set. Check the API startup logs — the backend logs
which variables are missing at startup. Verify your `.env` file or shell exports contain all four:
`KEYCLOAK_URL`, `KEYCLOAK_REALM`, `KEYCLOAK_CLIENT_ID`, and `KEYCLOAK_CLIENT_SECRET`.

**CORS errors in the browser after login**

Set `CORS_ORIGINS` to the exact origin of the frontend (e.g., `http://localhost:5173` for a local
Vite dev server, or `https://your-domain.example.com` for production). Multiple origins can be
comma-separated. The value must match the browser's `Origin` header exactly, including scheme and
port.

**Login redirects to the wrong URL after authentication**

Ensure the **Valid Redirect URIs** in the Keycloak client settings include the URL the backend is
running on. For production, replace `http://localhost:8001/*` with the production URL pattern.
