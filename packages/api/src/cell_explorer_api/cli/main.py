"""cell-explorer-chat CLI entry point."""

import os
import urllib.parse
import webbrowser
from datetime import datetime, timezone, timedelta
from typing import Any

import typer

from cell_explorer_api.cli.callback_server import CallbackTimeout, start_callback_server
from cell_explorer_api.cli.config import (
    AuthConfig,
    delete_auth_config,
    save_auth_config,
)

app = typer.Typer(add_completion=False, help="Cell Explorer CLI.")


@app.callback()
def _main() -> None:
    """Cell Explorer CLI."""


def _api_url() -> str:
    return os.environ.get("CELL_EXPLORER_API_URL", "http://localhost:8000")


def _decode_username(access_token: str) -> str:
    """Best-effort extract username from the JWT access token (no signature check).

    Authoritative validation happens inside the agent loop when the user is
    constructed from the Settings/KeycloakClient. Here we only want a display
    value for the login banner.
    """
    import jwt as _jwt

    try:
        claims = _jwt.decode(access_token, options={"verify_signature": False})
    except Exception:
        return "(unknown user)"
    return (
        claims.get("preferred_username")
        or claims.get("email")
        or claims.get("sub")
        or "(unknown user)"
    )


@app.command()
def login() -> None:
    """Open a browser, authenticate via Keycloak, and save auth.json."""
    api = _api_url()
    port, wait = start_callback_server(path="/callback", timeout_s=300)
    redirect_uri = f"http://localhost:{port}/callback"
    login_url = (
        f"{api}/api/auth/cli-login?"
        + urllib.parse.urlencode({"redirect_uri": redirect_uri})
    )

    typer.echo("Opening browser for login...")
    typer.echo(f"If it doesn't open automatically: {login_url}")
    webbrowser.open(login_url)

    try:
        tokens: dict[str, Any] = wait()
    except CallbackTimeout as exc:
        typer.echo(f"error: login timed out ({exc})", err=True)
        raise typer.Exit(code=1)

    now = datetime.now(timezone.utc)
    cfg = AuthConfig(
        api_url=api,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        access_expires_at=now + timedelta(seconds=int(tokens.get("expires_in", 300))),
        refresh_expires_at=now + timedelta(seconds=int(tokens.get("refresh_expires_in", 30 * 86400))),
        username=_decode_username(tokens["access_token"]),
    )
    save_auth_config(cfg)
    typer.echo(f"✓ Logged in as {cfg.username}")


@app.command()
def logout() -> None:
    """Delete local auth.json (idempotent)."""
    delete_auth_config()
    typer.echo("✓ Logged out.")


import asyncio
from dataclasses import dataclass

from sqlmodel import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource
from cell_explorer_api.services.access import user_can_access
from cell_explorer_api.cli.config import load_auth_config
from cell_explorer_api.cli.errors import NotLoggedInError, SessionExpiredError
from cell_explorer_api.cli.tokens import ensure_fresh_access_token

ERR_NOT_LOGGED_IN = 6
ERR_SESSION_EXPIRED = 6


@dataclass
class _User:
    username: str
    roles: list[str]


async def _load_user_from_auth(settings: Settings) -> _User:
    """Load auth.json, refresh if needed, decode the access token, return _User."""
    from cell_explorer_api.auth.keycloak import KeycloakClient

    cfg = load_auth_config()
    keycloak = KeycloakClient(settings)
    await keycloak.fetch_jwks()
    cfg = await ensure_fresh_access_token(cfg, keycloak)
    user_obj = keycloak.decode_token(cfg.access_token)
    return _User(username=user_obj.username, roles=list(user_obj.roles))


def _list_datasets_sync() -> list[dict]:
    """Return raw dataset rows as dicts (lists all, including ones the user can't access)."""
    settings = Settings()
    engine = create_async_engine(settings.effective_database_url, echo=False)

    async def _run():
        async with SQLModelAsyncSession(engine) as session:
            stmt = select(Dataset, Datasource).join(Datasource)
            result = await session.exec(stmt)
            rows = result.all()
            return [
                {
                    "slug": d.slug,
                    "name": d.name,
                    "is_public": d.is_public,
                    "required_roles": list(d.required_roles or []),
                }
                for d, _ in rows
            ]

    try:
        return asyncio.run(_run())
    finally:
        asyncio.run(engine.dispose())


def _load_user_sync() -> _User:
    settings = Settings()
    return asyncio.run(_load_user_from_auth(settings))


@app.command()
def datasets() -> None:
    """List datasets the authenticated user can access."""
    try:
        user = _load_user_sync()
    except NotLoggedInError:
        typer.echo("error: not logged in. Run 'cell-explorer-chat login'.")
        raise typer.Exit(code=ERR_NOT_LOGGED_IN)
    except SessionExpiredError:
        typer.echo("error: session expired. Run 'cell-explorer-chat login' again.")
        raise typer.Exit(code=ERR_SESSION_EXPIRED)

    rows = _list_datasets_sync()

    # Filter via the shared access helper
    class _DS:
        def __init__(self, row):
            self.is_public = row["is_public"]
            self.required_roles = row["required_roles"]

    visible = [row for row in rows if user_can_access(_DS(row), user=user)]

    if not visible:
        typer.echo("(no accessible datasets)")
        return

    typer.echo(f"{'slug':<12} {'name':<30} {'access'}")
    for row in visible:
        access = "public" if row["is_public"] else f"requires: {', '.join(row['required_roles'])}"
        typer.echo(f"{row['slug']:<12} {row['name']:<30} {access}")


if __name__ == "__main__":
    app()
