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
