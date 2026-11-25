"""Token verifier: user tokens are traded for API tokens at the exchange endpoint."""

import hashlib
from urllib.parse import parse_qs

import httpx
import pytest

from cell_explorer_mcp import auth
from cell_explorer_mcp.auth import CellExplorerTokenVerifier

URL = "http://api.test/api/oauth/token-exchange"


def make_verifier(status=200, expires_in=600, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "access_token": "api-tok",
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "token_type": "Bearer",
                "expires_in": expires_in,
                "subject": "user-1",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CellExplorerTokenVerifier(
        exchange_url=URL, mcp_key="mcp-key", client=client
    )


async def test_valid_token_yields_api_token():
    v = make_verifier()
    at = await v.verify_token("user-tok")
    assert at is not None
    assert at.token == "user-tok"
    assert at.claims == {"api_token": "api-tok", "sub": "user-1"}


async def test_rejected_token_is_none():
    assert await make_verifier(status=400).verify_token("bad") is None
    assert await make_verifier(status=401).verify_token("bad") is None


async def test_network_error_is_none():
    def handler(request):
        raise httpx.ConnectError("down")

    v = CellExplorerTokenVerifier(
        exchange_url=URL,
        mcp_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert await v.verify_token("t") is None


async def test_mcp_key_sent_as_bearer():
    calls = []
    await make_verifier(calls=calls).verify_token("user-tok")
    req = calls[0]
    assert str(req.url) == URL
    assert req.headers["authorization"] == "Bearer mcp-key"
    form = parse_qs(req.content.decode())
    assert form["subject_token"] == ["user-tok"]
    assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:token-exchange"]
    assert form["subject_token_type"] == [
        "urn:ietf:params:oauth:token-type:access_token"
    ]


async def test_exchange_cached_until_near_expiry(monkeypatch):
    calls = []
    v = make_verifier(calls=calls, expires_in=600)
    now = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: now[0])
    await v.verify_token("t")
    now[0] += 500
    await v.verify_token("t")
    assert len(calls) == 1
    now[0] += 80  # within 30 s of expiry
    await v.verify_token("t")
    assert len(calls) == 2


async def test_failures_not_cached():
    calls = []
    v = make_verifier(status=400, calls=calls)
    await v.verify_token("t")
    await v.verify_token("t")
    assert len(calls) == 2


async def test_cache_keyed_by_hash_and_bounded(monkeypatch):
    v = make_verifier()
    monkeypatch.setattr(auth, "MAX_CACHE_ENTRIES", 3)
    for i in range(10):
        await v.verify_token(f"t{i}")
    assert len(v._cache) <= 3
    assert "t9" not in v._cache
    assert hashlib.sha256(b"t9").hexdigest() in v._cache


def test_current_helpers(monkeypatch):
    from fastmcp.server.auth.auth import AccessToken

    tok = AccessToken(
        token="u", client_id="c", scopes=[], claims={"api_token": "a", "sub": "s"}
    )
    monkeypatch.setattr(auth, "get_access_token", lambda: tok)
    assert auth.current_api_token() == "a"
    assert auth.current_user_sub() == "s"
    monkeypatch.setattr(auth, "get_access_token", lambda: None)
    with pytest.raises(RuntimeError):
        auth.current_api_token()
    with pytest.raises(RuntimeError):
        auth.current_user_sub()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>proxy</html>"),
        httpx.Response(200, json={"access_token": "a", "expires_in": 600}),
        httpx.Response(200, json=["not", "a", "dict"]),
    ],
)
async def test_malformed_200_is_none_and_not_cached(response):
    v = CellExplorerTokenVerifier(
        exchange_url=URL,
        mcp_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response)),
    )
    assert await v.verify_token("t") is None
    assert v._cache == {}


async def test_refresh_at_capacity_keeps_other_entries(monkeypatch):
    v = make_verifier()
    monkeypatch.setattr(auth, "MAX_CACHE_ENTRIES", 2)
    await v.verify_token("a")
    await v.verify_token("b")
    v._store(hashlib.sha256(b"b").hexdigest(), v._cache[hashlib.sha256(b"b").hexdigest()])
    assert len(v._cache) == 2
