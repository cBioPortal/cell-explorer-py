"""Auth config stored at ~/.config/cell-explorer/auth.json (0600)."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from cell_explorer_api.cli.errors import NotLoggedInError


@dataclass
class AuthConfig:
    api_url: str
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    username: str


def auth_config_path() -> Path:
    return Path.home() / ".config" / "cell-explorer" / "auth.json"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def save_auth_config(cfg: AuthConfig) -> None:
    path = auth_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    data["access_expires_at"] = _iso(cfg.access_expires_at)
    data["refresh_expires_at"] = _iso(cfg.refresh_expires_at)
    text = json.dumps(data, indent=2) + "\n"
    # Write to a tempfile and chmod before rename to avoid a brief 0644 window.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_auth_config() -> AuthConfig:
    path = auth_config_path()
    if not path.is_file():
        raise NotLoggedInError(f"no auth.json at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise NotLoggedInError(f"could not read {path}: {exc}") from exc
    try:
        return AuthConfig(
            api_url=data["api_url"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            access_expires_at=_parse_iso(data["access_expires_at"]),
            refresh_expires_at=_parse_iso(data["refresh_expires_at"]),
            username=data["username"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise NotLoggedInError(f"auth.json at {path} is malformed: {exc}") from exc


def delete_auth_config() -> None:
    path = auth_config_path()
    if path.is_file():
        path.unlink()
