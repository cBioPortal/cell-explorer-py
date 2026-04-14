"""Tests for zarr-auth-proxy settings."""

from pathlib import Path

import pytest

from zarr_auth_proxy.config import Settings


def test_settings_requires_public_key_file(tmp_path):
    key_file = tmp_path / "public.pem"
    key_file.write_text("fake-key")
    settings = Settings(public_key_file=key_file, data_dir=tmp_path)
    assert settings.public_key_file == key_file


def test_settings_requires_data_dir(tmp_path):
    key_file = tmp_path / "public.pem"
    key_file.write_text("fake-key")
    settings = Settings(public_key_file=key_file, data_dir=tmp_path)
    assert settings.data_dir == tmp_path


def test_settings_cors_origins_parsing(tmp_path):
    key_file = tmp_path / "public.pem"
    key_file.write_text("fake-key")
    settings = Settings(
        public_key_file=key_file,
        data_dir=tmp_path,
        cors_origins="http://localhost:8001,https://example.com",
    )
    assert settings.cors_origin_list == ["http://localhost:8001", "https://example.com"]


def test_settings_cors_origins_empty(tmp_path):
    key_file = tmp_path / "public.pem"
    key_file.write_text("fake-key")
    settings = Settings(public_key_file=key_file, data_dir=tmp_path)
    assert settings.cors_origin_list == []


def test_settings_port_default(tmp_path):
    key_file = tmp_path / "public.pem"
    key_file.write_text("fake-key")
    settings = Settings(public_key_file=key_file, data_dir=tmp_path)
    assert settings.port == 8000
