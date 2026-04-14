"""Zarr auth proxy settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Zarr auth proxy configuration.

    All fields can be set via environment variables (case-insensitive).
    """

    public_key_file: Path
    data_dir: Path
    cors_origins: str = ""
    port: int = 8000

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
