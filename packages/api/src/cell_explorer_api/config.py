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

    # App data directory
    app_data_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"
    log_rotation_interval: str = "daily"
    log_backup_count: int = 30
    log_filename: str = "cell-explorer.log"

    # Auth (all optional — auth disabled when keycloak_url is not set)
    keycloak_url: str | None = None
    keycloak_realm: str | None = None
    keycloak_client_id: str | None = None
    keycloak_client_secret: str | None = None
    keycloak_idp_hint: str | None = None
    cors_origins: str = ""

    # Session cookie lifetimes (seconds). Defaults match a typical Keycloak
    # cell-explorer realm: 5m access, 24h refresh. Set ACCESS_COOKIE_MAX_AGE
    # / REFRESH_COOKIE_MAX_AGE in env to override; tune the refresh value to
    # be <= the realm's ssoSessionMaxLifespan or refresh will fail early.
    access_cookie_max_age: int = 300
    refresh_cookie_max_age: int = 86400

    # Database
    database_url: str | None = None

    # Admin
    admin_api_key: str | None = None

    # Chat (LLM)
    anthropic_api_key: str | None = None
    # Optional Keycloak role required for chat access. None = any
    # authenticated user can chat (subject to per-dataset chat_enabled).
    chat_required_role: str | None = None

    # CLI integration
    cli_state_secret: str | None = None

    @property
    def log_dir(self) -> Path:
        """Directory for log files."""
        return self.app_data_dir / "logs"

    @property
    def log_rotation_when(self) -> str:
        """Map human-readable interval to TimedRotatingFileHandler 'when' parameter."""
        mapping = {"daily": "midnight", "hourly": "h", "weekly": "w0"}
        return mapping.get(self.log_rotation_interval, "midnight")

    @property
    def effective_database_url(self) -> str:
        """Database URL, defaulting to SQLite in app_data_dir if not explicitly set."""
        if self.database_url is not None:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.app_data_dir / 'cell_explorer.db'}"

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

    @property
    def admin_enabled(self) -> bool:
        """Admin API is enabled when ADMIN_API_KEY is set."""
        return self.admin_api_key is not None

    @property
    def chat_enabled(self) -> bool:
        """Chat is enabled when ANTHROPIC_API_KEY is set."""
        return self.anthropic_api_key is not None

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
