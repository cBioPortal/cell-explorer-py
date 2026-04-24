import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from cell_explorer_api.cli.config import (
    AuthConfig,
    auth_config_path,
    load_auth_config,
    save_auth_config,
)
from cell_explorer_api.cli.errors import NotLoggedInError


def test_auth_config_path_uses_xdg_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert auth_config_path() == tmp_path / ".config" / "cell-explorer" / "auth.json"


def test_load_missing_raises_not_logged_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(NotLoggedInError):
        load_auth_config()


def test_save_and_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="ACC",
        refresh_token="REF",
        access_expires_at=now + timedelta(minutes=5),
        refresh_expires_at=now + timedelta(days=30),
        username="jason@example.com",
    )
    save_auth_config(cfg)

    loaded = load_auth_config()
    assert loaded.api_url == cfg.api_url
    assert loaded.access_token == "ACC"
    assert loaded.refresh_token == "REF"
    assert loaded.username == "jason@example.com"
    assert loaded.access_expires_at == cfg.access_expires_at
    assert loaded.refresh_expires_at == cfg.refresh_expires_at


def test_save_file_is_mode_0600(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc),
        refresh_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        username="x",
    )
    save_auth_config(cfg)

    path = auth_config_path()
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600


def test_save_creates_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = AuthConfig(
        api_url="http://localhost:8000",
        access_token="A", refresh_token="R",
        access_expires_at=datetime.now(timezone.utc),
        refresh_expires_at=datetime.now(timezone.utc),
        username="x",
    )
    save_auth_config(cfg)
    assert (tmp_path / ".config" / "cell-explorer").is_dir()


def test_load_malformed_json_raises_not_logged_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = auth_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json")
    with pytest.raises(NotLoggedInError):
        load_auth_config()
