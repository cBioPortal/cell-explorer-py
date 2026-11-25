"""The HTTP app: the open address, plus the signed-in address when an MCP key is set."""

import json
from urllib.parse import urlsplit

import fastmcp
import pytest
from fastmcp.server.auth.auth import AccessToken
from starlette.testclient import TestClient

from cell_explorer_mcp import app as app_module
from cell_explorer_mcp import queries, server

from conftest import stub_catalog

PUBLIC_URL = "https://mcp.test/signed-in/mcp"
ACCEPT = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def env(monkeypatch):
    for name in ("CELL_EXPLORER_MCP_KEY", "MCP_PUBLIC_URL", "MCP_SIGNED_IN_HTTP_PATH", "MCP_HTTP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queries, "UI_URL", "https://ce.test")
    monkeypatch.setattr(server, "catalog", stub_catalog("http://stores.test"))
    return monkeypatch


@pytest.fixture
def with_key(env):
    env.setenv("CELL_EXPLORER_MCP_KEY", "mcp-key")
    env.setenv("MCP_PUBLIC_URL", PUBLIC_URL)
    return env


def _rpc(client, path, method, params=None, headers=None, rid=1):
    body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
    return client.post(path, json=body, headers={**ACCEPT, **(headers or {})})


def _result(resp):
    assert resp.status_code == 200, resp.text
    if resp.headers["content-type"].startswith("application/json"):
        return resp.json()["result"]
    data = [line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")]
    return json.loads(data[-1])["result"]


def _session(client, path, headers=None):
    """Initialize an MCP session; returns the headers for later requests."""
    init = _rpc(client, path, "initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    }, headers)
    _result(init)
    session = {**(headers or {})}
    if "mcp-session-id" in init.headers:
        session["mcp-session-id"] = init.headers["mcp-session-id"]
    client.post(path, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers={**ACCEPT, **session})
    return session


def _tool_names(client, path, headers=None):
    session = _session(client, path, headers)
    tools = _result(_rpc(client, path, "tools/list", headers=session, rid=2))["tools"]
    return {t["name"] for t in tools}


def _call(client, path, tool, args, headers=None):
    session = _session(client, path, headers)
    return _result(_rpc(client, path, "tools/call", {"name": tool, "arguments": args}, session, rid=3))


def test_without_key_only_open_address(env):
    with TestClient(app_module.build_http_app()) as client:
        assert client.post("/signed-in/mcp", json={}, headers=ACCEPT).status_code == 404
        assert "list_datasets" in _tool_names(client, "/mcp")


def test_signed_in_address_requires_token(with_key):
    with TestClient(app_module.build_http_app()) as client:
        resp = _rpc(client, "/signed-in/mcp", "tools/list")
        assert resp.status_code == 401
        assert "resource_metadata=" in resp.headers["www-authenticate"]


def test_protected_resource_metadata_names_cell_explorer(with_key):
    with TestClient(app_module.build_http_app()) as client:
        challenge = _rpc(client, "/signed-in/mcp", "tools/list").headers["www-authenticate"]
        metadata_url = challenge.split('resource_metadata="', 1)[1].split('"', 1)[0]
        assert metadata_url == "https://mcp.test/.well-known/oauth-protected-resource/signed-in/mcp"
        metadata = client.get(urlsplit(metadata_url).path).json()
        assert metadata["authorization_servers"] == ["https://ce.test/api/oauth"]
        assert metadata["resource"] == PUBLIC_URL


def test_open_address_unchanged_with_key(with_key):
    with TestClient(app_module.build_http_app()) as client:
        # Anonymous, and a stray bearer token is not checked on the open address.
        assert "list_datasets" in _tool_names(client, "/mcp")
        assert "list_datasets" in _tool_names(client, "/mcp", {"Authorization": "Bearer junk"})
        listed = _call(client, "/mcp", "list_datasets", {})["structuredContent"]
        assert {d["slug"] for d in listed["datasets"]} == {"strata-tiny", "pbmc3k"}
        refused = _call(client, "/mcp", "get_dataset_info", {"dataset": "secret"})
        assert refused["isError"] is True
        assert "is not public" in refused["content"][0]["text"]


def test_key_without_public_url_fails_fast(env):
    env.setenv("CELL_EXPLORER_MCP_KEY", "mcp-key")
    with pytest.raises(RuntimeError, match="MCP_PUBLIC_URL"):
        app_module.build_http_app()


def test_public_url_must_end_with_signed_in_path(env):
    env.setenv("CELL_EXPLORER_MCP_KEY", "mcp-key")
    env.setenv("MCP_PUBLIC_URL", "https://mcp.test/mcp")
    with pytest.raises(RuntimeError, match="MCP_SIGNED_IN_HTTP_PATH"):
        app_module.build_http_app()


class _FakeVerifier(fastmcp.server.auth.auth.TokenVerifier):
    def __init__(self, *, exchange_url, mcp_key):
        super().__init__()
        self.seen = (exchange_url, mcp_key)

    async def verify_token(self, token):
        if token != "good":
            return None
        return AccessToken(token=token, client_id="cell-explorer", scopes=[],
                           claims={"api_token": "api", "sub": "u1"})


def test_signed_in_address_serves_tools_with_valid_token(with_key):
    with_key.setenv("CELL_EXPLORER_API_URL", "http://api.internal/api/")
    verifiers = []
    with_key.setattr(app_module, "CellExplorerTokenVerifier",
                     lambda **kw: verifiers.append(_FakeVerifier(**kw)) or verifiers[-1])
    with TestClient(app_module.build_http_app()) as client:
        names = _tool_names(client, "/signed-in/mcp", {"Authorization": "Bearer good"})
        assert names == _tool_names(client, "/mcp")
        assert _rpc(client, "/signed-in/mcp", "tools/list",
                    headers={"Authorization": "Bearer bad"}).status_code == 401
    assert verifiers[0].seen == ("http://api.internal/api/oauth/token-exchange", "mcp-key")


def test_stateless_setting_applies_to_both_addresses(with_key):
    with_key.setattr(fastmcp.settings, "stateless_http", True)
    with_key.setattr(app_module, "CellExplorerTokenVerifier", _FakeVerifier)
    with TestClient(app_module.build_http_app()) as client:
        # Stateless: no session, so tools/list works without initialize.
        assert _result(_rpc(client, "/mcp", "tools/list"))["tools"]
        assert _result(_rpc(client, "/signed-in/mcp", "tools/list",
                            headers={"Authorization": "Bearer good"}))["tools"]
