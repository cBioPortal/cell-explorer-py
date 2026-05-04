"""Access-token refresh logic for the CLI."""

from datetime import datetime, timezone, timedelta
from typing import Any

from cell_explorer_api.cli.config import AuthConfig
from cell_explorer_api.cli.errors import SessionExpiredError

_REFRESH_THRESHOLD_S = 60


def _needs_refresh(cfg: AuthConfig) -> bool:
    now = datetime.now(timezone.utc)
    return cfg.access_expires_at <= now + timedelta(seconds=_REFRESH_THRESHOLD_S)


async def ensure_fresh_access_token(cfg: AuthConfig, keycloak: Any) -> AuthConfig:
    """If the access token is near expiry, refresh it via Keycloak.

    Returns an AuthConfig reflecting the current (possibly-refreshed) tokens.
    Raises SessionExpiredError if refresh fails.
    """
    if not _needs_refresh(cfg):
        return cfg

    try:
        tokens = await keycloak.refresh_token(cfg.refresh_token)
    except Exception as exc:
        raise SessionExpiredError(str(exc)) from exc

    now = datetime.now(timezone.utc)
    return AuthConfig(
        api_url=cfg.api_url,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        access_expires_at=now + timedelta(seconds=int(tokens.get("expires_in", 300))),
        refresh_expires_at=now + timedelta(seconds=int(tokens.get("refresh_expires_in", 30 * 86400))),
        username=cfg.username,
    )
