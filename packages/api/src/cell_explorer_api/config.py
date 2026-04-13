"""Application settings loaded from environment variables."""

import logging
import subprocess
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _detect_git_sha() -> str | None:
    """Try to read the current git SHA. Returns None if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


class Settings(BaseSettings):
    """Cell Explorer API settings.

    All fields can be set via environment variables (case-insensitive).
    """

    static_dir: Path | None = None
    environment: str = "development"
    git_sha: str | None = None

    # Auth (all optional — auth disabled when keycloak_url is not set)
    keycloak_url: str | None = None
    keycloak_realm: str | None = None
    keycloak_client_id: str | None = None
    keycloak_client_secret: str | None = None
    cors_origins: str = ""

    @property
    def auth_enabled(self) -> bool:
        """Auth is enabled when all required Keycloak fields are set."""
        return all([
            self.keycloak_url,
            self.keycloak_realm,
            self.keycloak_client_id,
            self.keycloak_client_secret,
        ])

    @property
    def oidc_issuer_url(self) -> str | None:
        """Keycloak OIDC issuer URL, constructed from base URL and realm."""
        if not self.keycloak_url or not self.keycloak_realm:
            return None
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _autodetect_git_sha(self) -> "Settings":
        if self.git_sha is None:
            self.git_sha = _detect_git_sha()
        return self


def validate_static_dir(path: Path) -> Path | None:
    """Validate that a static directory exists and contains index.html.

    Returns the path if valid, None otherwise.
    """
    if not path.is_dir():
        logger.warning("STATIC_DIR %s does not exist", path)
        return None
    if not (path / "index.html").is_file():
        logger.warning("STATIC_DIR %s has no index.html", path)
        return None
    return path
