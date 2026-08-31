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

    # GA4 measurement ID (G-XXXXXXXXXX), served to the frontend by /api/info.
    # Set per environment rather than baked into the frontend build, because one
    # image is deployed to several environments. Unset means untracked: the
    # frontend loads no analytics script at all. Not a secret — a measurement id
    # is public in the page source of every site that uses one.
    google_analytics_id: str | None = None

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

    # Generic OIDC (provider-agnostic). auth_provider selects preset defaults;
    # KEYCLOAK_* remain the zero-config path for auth_provider="keycloak".
    auth_provider: str = "keycloak"  # keycloak | entra | oidc
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_scopes: str = "openid profile email"
    oidc_roles_claims: str | None = None
    oidc_audience: str | None = None

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
    def resolved_client_id(self) -> str | None:
        return self.oidc_client_id or self.keycloak_client_id

    @property
    def resolved_client_secret(self) -> str | None:
        return self.oidc_client_secret or self.keycloak_client_secret

    @property
    def resolved_issuer(self) -> str | None:
        """OIDC issuer URL. Keycloak derives it from url+realm; others use OIDC_ISSUER."""
        if self.oidc_issuer:
            return self.oidc_issuer
        if self.auth_provider == "keycloak" and self.keycloak_url and self.keycloak_realm:
            return f"{self.keycloak_url}/realms/{self.keycloak_realm}"
        return None

    @property
    def discovery_url(self) -> str | None:
        issuer = self.resolved_issuer
        return f"{issuer}/.well-known/openid-configuration" if issuer else None

    @property
    def resolved_audience(self) -> str | None:
        return self.oidc_audience or self.resolved_client_id

    @property
    def resolved_scopes(self) -> str:
        scopes = self.oidc_scopes
        if self.auth_provider == "entra" and "offline_access" not in scopes.split():
            scopes = f"{scopes} offline_access"
        return scopes

    @property
    def resolved_idp_hint(self) -> str | None:
        return self.keycloak_idp_hint if self.auth_provider == "keycloak" else None

    @property
    def resolved_roles_claims(self) -> list[str]:
        """Dotted claim-paths whose role lists are merged into User.roles."""
        if self.oidc_roles_claims:
            return [p.strip() for p in self.oidc_roles_claims.split(",") if p.strip()]
        if self.auth_provider == "keycloak":
            return ["realm_access.roles", f"resource_access.{self.resolved_client_id}.roles"]
        if self.auth_provider == "entra":
            return ["roles"]
        return []

    @property
    def auth_enabled(self) -> bool:
        """Auth is enabled when issuer + client id + secret resolve."""
        return all([self.resolved_issuer, self.resolved_client_id, self.resolved_client_secret])

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
