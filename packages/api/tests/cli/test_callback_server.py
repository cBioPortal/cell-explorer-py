import json
import urllib.error
import urllib.request

import pytest

from cell_explorer_api.cli.callback_server import CallbackTimeout, start_callback_server


def test_callback_server_returns_posted_tokens():
    port, wait = start_callback_server(path="/callback", timeout_s=5)
    body = json.dumps({
        "access_token": "ACC",
        "refresh_token": "REF",
        "expires_in": 300,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/callback",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 204

    tokens = wait()
    assert tokens == {"access_token": "ACC", "refresh_token": "REF", "expires_in": 300}


def test_callback_server_times_out():
    port, wait = start_callback_server(path="/callback", timeout_s=0.2)
    with pytest.raises(CallbackTimeout):
        wait()


def test_callback_server_ignores_wrong_path():
    port, wait = start_callback_server(path="/callback", timeout_s=1)
    req = urllib.request.Request(
        f"http://localhost:{port}/bogus",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=2)
    assert exc_info.value.code == 404

    with pytest.raises(CallbackTimeout):
        wait()
