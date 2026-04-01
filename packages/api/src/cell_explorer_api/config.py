"""Application settings loaded from environment variables."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Cell Explorer API settings.

    All fields can be set via environment variables (case-insensitive).
    """

    static_dir: Path | None = None


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
