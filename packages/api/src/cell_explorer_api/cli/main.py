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


if __name__ == "__main__":
    app()
