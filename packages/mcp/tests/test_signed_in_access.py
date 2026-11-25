"""The signed-in address acts as the signed-in user: their catalog, their
private datasets, and private stores no other user is served."""

import contextlib

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.auth import AccessToken
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from cell_explorer_mcp import server
from cell_explorer_mcp.zarr_cache import DatasetCache, open_dataset

from conftest import PRIVATE_STORE_COOKIES, PRIVATE_STORE_TOKEN, cloudfront_set_cookies, stub_catalog


class SpyOpener:
    def __init__(self):
        self.opened: list[tuple[str, str, dict | None]] = []
        self.stores: list = []

    async def __call__(self, slug, url, headers=None):
        self.opened.append((slug, url, headers))
        self.stores.append(await open_dataset(slug, url, headers=headers))
        return self.stores[-1]


@pytest.fixture
def api_log():
    return []


@pytest.fixture
def api():
    return {}


@pytest.fixture
def opener():
    return SpyOpener()


@pytest.fixture
def as_user(fixture_server, monkeypatch, api, api_log, opener):
    """as_user("alice") connects to the signed-in server as alice.

    The in-memory transport skips token verification, so the user is put
    where the HTTP auth middleware would: the request's auth context.
    """
    monkeypatch.setattr(server, "catalog", stub_catalog(fixture_server, api=api, log=api_log))
    monkeypatch.setattr(server, "cache", DatasetCache(opener=opener))
    signed_in = server.build_server(signed_in=True)

    @contextlib.asynccontextmanager
    async def connect(name, mcp=signed_in):
        user = AuthenticatedUser(AccessToken(
            token=f"mcp-{name}", client_id="cell-explorer", scopes=[],
            claims={"api_token": f"api-{name}", "sub": f"sub-{name}"},
        ))
        reset = auth_context_var.set(user)
        try:
            async with Client(mcp) as client:
                yield client
        finally:
            auth_context_var.reset(reset)

    return connect


def _access_callers(api_log):
    return [bearer for method, _, bearer in api_log if method == "POST"]


def _slugs(result):
    return {d["slug"] for d in result.data["datasets"]}


async def test_signed_in_lists_users_private_datasets(as_user):
    async with as_user("alice") as client:
        datasets = await client.call_tool("list_datasets", {})
        collections = (await client.call_tool("list_collections", {})).data
    assert _slugs(datasets) == {"strata-tiny", "pbmc3k", "secret", "secret-cookies"}
    assert "private-studies" in {c["slug"] for c in collections["collections"]}

    async with as_user("carol") as client:
        datasets = await client.call_tool("list_datasets", {})
        collections = (await client.call_tool("list_collections", {})).data
        with pytest.raises(ToolError, match="Unknown dataset"):
            await client.call_tool("get_dataset_info", {"dataset": "secret"})
    assert _slugs(datasets) == {"strata-tiny", "pbmc3k"}
    assert "private-studies" not in {c["slug"] for c in collections["collections"]}

    # A user's listing never reaches the shared anonymous one.
    anonymous = await server.catalog.list_datasets()
    assert "secret-cookies" not in {d["slug"] for d in anonymous}


async def test_private_dataset_opened_with_credential_headers(as_user, opener, fixture_server):
    url = f"{fixture_server}/private/strata-tiny.zarr"
    with pytest.raises(Exception):
        await open_dataset("secret", url)  # the private store refuses anonymous reads

    async with as_user("alice") as client:
        for slug in ("secret", "secret-cookies"):
            info = (await client.call_tool("get_dataset_info", {"dataset": slug})).data
            assert info["n_cells"] == 50
    cookie = "; ".join(f"{k}={v}" for k, v in PRIVATE_STORE_COOKIES.items())
    assert opener.opened == [
        ("secret", url, {"Authorization": f"Bearer {PRIVATE_STORE_TOKEN}"}),
        ("secret-cookies", url, {"Cookie": cookie}),
    ]


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_access_denied_is_tool_error(as_user, api, opener, status):
    api["access_status"] = status
    async with as_user("alice") as client:
        with pytest.raises(ToolError, match=r"You don't have access to dataset 'secret'\.$"):
            await client.call_tool("get_dataset_info", {"dataset": "secret"})
    assert opener.opened == []


async def test_set_cookie_credential_keeps_only_cloudfront_cookies(as_user, api, opener):
    api["set_cookies"] = cloudfront_set_cookies() + [("set-cookie", "session=abc; Path=/; HttpOnly")]
    async with as_user("alice") as client:
        await client.call_tool("get_dataset_info", {"dataset": "secret-cookies"})
    cookie = "; ".join(f"{k}={v}" for k, v in PRIVATE_STORE_COOKIES.items())
    assert opener.opened[0][2] == {"Cookie": cookie}


async def test_signed_cookie_credential_without_cookies_is_tool_error(as_user, api, opener):
    api["set_cookies"] = [("set-cookie", "session=abc; Path=/; HttpOnly")]
    async with as_user("alice") as client:
        with pytest.raises(ToolError, match=r"Could not open dataset 'secret-cookies'.*no CloudFront"):
            await client.call_tool("get_dataset_info", {"dataset": "secret-cookies"})
    assert opener.opened == []


async def test_access_checked_on_every_private_call(as_user, api, api_log, opener):
    async with as_user("alice") as client:
        await client.call_tool("get_dataset_info", {"dataset": "secret"})
        await client.call_tool("search_genes", {"dataset": "secret", "query": "gene1"})
        # Revoked: her store is still cached, but /access now refuses her.
        api["access_status"] = 403
        with pytest.raises(ToolError, match=r"You don't have access to dataset 'secret'\.$"):
            await client.call_tool("search_genes", {"dataset": "secret", "query": "gene1"})
    assert ("secret", "sub-alice") in server.cache
    assert _access_callers(api_log) == ["api-alice"] * 3
    assert len(opener.opened) == 1  # the second call reused her store


async def test_private_store_not_served_to_other_user(as_user, api_log, opener):
    async with as_user("alice") as client:
        await client.call_tool("get_dataset_info", {"dataset": "secret"})
        await client.call_tool("search_genes", {"dataset": "secret", "query": "gene1"})
    assert ("secret", "sub-alice") in server.cache

    # Within alice's credential lifetime: dave's /access is refused, so he
    # gets an error, not alice's open store.
    async with as_user("dave") as client:
        with pytest.raises(ToolError, match="You don't have access"):
            await client.call_tool("gene_expression", {"dataset": "secret", "gene": "gene1"})

    async with as_user("bob") as client:
        await client.call_tool("gene_expression", {"dataset": "secret", "gene": "gene1"})

    # Every call asked /access as its own user; alice's second call reused her
    # store, and bob got a store of his own.
    assert _access_callers(api_log) == ["api-alice", "api-alice", "api-dave", "api-bob"]
    assert [slug for slug, _, _ in opener.opened] == ["secret", "secret"]
    alice_store, bob_store = opener.stores
    assert bob_store is not alice_store
    assert ("secret", "sub-bob") in server.cache
    assert ("secret", "sub-dave") not in server.cache


async def test_private_store_reopened_after_credential_ttl(as_user, api_log, opener, monkeypatch):
    assert server.PRIVATE_STORE_TTL_SECONDS == 1800 - 60
    monkeypatch.setattr(server, "PRIVATE_STORE_TTL_SECONDS", 0)
    async with as_user("alice") as client:
        await client.call_tool("get_dataset_info", {"dataset": "secret"})
        await client.call_tool("get_dataset_info", {"dataset": "secret"})
    assert _access_callers(api_log) == ["api-alice", "api-alice"]
    assert len(opener.opened) == 2


async def test_public_dataset_uses_shared_store(as_user, api_log, opener):
    async with as_user("alice") as client:
        await client.call_tool("get_dataset_info", {"dataset": "strata-tiny"})
    async with as_user("alice", mcp=server.mcp) as client:
        await client.call_tool("get_dataset_info", {"dataset": "strata-tiny"})
    assert _access_callers(api_log) == []
    assert [(slug, headers) for slug, _, headers in opener.opened] == [("strata-tiny", None)]


async def test_open_address_still_refuses_private(as_user, api_log, opener):
    # Even with a user in the auth context, the open server stays anonymous.
    async with as_user("alice", mcp=server.mcp) as client:
        assert _slugs(await client.call_tool("list_datasets", {})) == {"strata-tiny", "pbmc3k"}
        with pytest.raises(ToolError, match="is not public"):
            await client.call_tool("get_dataset_info", {"dataset": "secret"})
        with pytest.raises(ToolError, match="Unknown dataset"):
            await client.call_tool("get_dataset_info", {"dataset": "secret-cookies"})
    assert all(bearer is None for _, _, bearer in api_log)
    assert opener.opened == []


async def test_list_datasets_description_per_address(as_user):
    async with as_user("alice") as client:
        signed_in = {t.name: t.description for t in await client.list_tools()}
    async with as_user("alice", mcp=server.mcp) as client:
        open_ = {t.name: t.description for t in await client.list_tools()}
    assert open_["list_datasets"] == "List public Cell Explorer datasets with their size and collection."
    assert signed_in["list_datasets"].startswith("List the Cell Explorer datasets you can access")
    assert {k: v for k, v in signed_in.items() if k != "list_datasets"} == {
        k: v for k, v in open_.items() if k != "list_datasets"
    }


@pytest.mark.parametrize("sub", ["", None])
async def test_private_store_refused_without_user_sub(fixture_server, monkeypatch, api, api_log, opener, sub):
    monkeypatch.setattr(server, "catalog", stub_catalog(fixture_server, api=api, log=api_log))
    monkeypatch.setattr(server, "cache", DatasetCache(opener=opener))
    user = AuthenticatedUser(AccessToken(
        token="mcp-alice", client_id="cell-explorer", scopes=[], claims={"api_token": "api-alice", "sub": sub},
    ))
    reset = auth_context_var.set(user)
    try:
        async with Client(server.build_server(signed_in=True)) as client:
            with pytest.raises(ToolError, match="signed-in user"):
                await client.call_tool("get_dataset_info", {"dataset": "secret"})
    finally:
        auth_context_var.reset(reset)
    assert opener.opened == []
