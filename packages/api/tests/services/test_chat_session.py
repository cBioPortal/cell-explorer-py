from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cell_explorer_agent import FakeLLMClient, Script
from cell_explorer_agent.config import AgentConfig

from cell_explorer_api.services.chat_session import (
    AccessDeniedError,
    ChatDisabledError,
    ChatSessionError,
    DatasetNotFoundError,
    make_chat_agent,
)


class _FakeUser:
    def __init__(self, roles: list[str]) -> None:
        self.roles = roles


def _make_db_row(dataset=None, datasource=None):
    """Produce a (dataset, datasource) tuple."""
    return (dataset, datasource)


async def _mk_db_session(row):
    """Async-mock DB session whose .exec().first() returns the given row."""
    result = MagicMock()
    result.first.return_value = row
    db = MagicMock()
    db.exec = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_dataset_not_found_raises():
    db = await _mk_db_session(None)
    settings = MagicMock()
    user = _FakeUser(roles=[])
    llm = FakeLLMClient(scripts=[])

    with pytest.raises(DatasetNotFoundError, match="nope"):
        await make_chat_agent(
            user=user, dataset_slug="nope", db=db, settings=settings, llm=llm
        )


@pytest.mark.asyncio
async def test_public_dataset_returns_agent_without_minting():
    dataset = MagicMock(slug="pbmc3k", name="PBMC 3k", description="",
                         is_public=True, required_roles=[], chat_enabled=True, path="pbmc3k.zarr")
    datasource = MagicMock(base_url="https://example.com", type="HTTP_TOKEN",
                            credential_ref=None)
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=[])
    llm = FakeLLMClient(scripts=[])

    # Patch ZarrStore.open + AnnDataStore.open to return stubs.
    fake_anndata = MagicMock()
    fake_anndata.n_obs = 10
    fake_anndata.n_vars = 20
    fake_anndata.obsm_keys = []
    fake_anndata.obs_columns = []

    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS, \
         patch("cell_explorer_api.services.chat_session.mint_credentials") as mock_mint:
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())

        agent = await make_chat_agent(
            user=user, dataset_slug="pbmc3k", db=db, settings=settings, llm=llm
        )

        # Public path never mints credentials
        mock_mint.assert_not_called()
        # ZarrStore.open called with empty headers for public
        MockZS.open.assert_awaited_once()
        _, kwargs = MockZS.open.call_args
        assert kwargs["headers"] == {}
        assert agent is not None


@pytest.mark.asyncio
async def test_private_dataset_without_role_raises_403():
    dataset = MagicMock(slug="brca", is_public=False,
                         required_roles=["researcher"], chat_enabled=True)
    datasource = MagicMock()
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=["guest"])
    llm = FakeLLMClient(scripts=[])

    with pytest.raises(AccessDeniedError, match="researcher"):
        await make_chat_agent(
            user=user, dataset_slug="brca", db=db, settings=settings, llm=llm
        )


from cell_explorer_api.services.chat_session import (
    CredentialMintError,
    _credential_to_headers,
)
from cell_explorer_api.services.credentials import CredentialError


def test_credential_to_headers_public():
    assert _credential_to_headers({"credential_type": "public"}) == {}


def test_credential_to_headers_bearer_token():
    headers = _credential_to_headers({
        "credential_type": "bearer_token",
        "token": "abc123",
    })
    assert headers == {"Authorization": "Bearer abc123"}


def test_credential_to_headers_signed_cookies():
    headers = _credential_to_headers({
        "credential_type": "signed_cookies",
        "cookies": {"CloudFront-Policy": "p", "CloudFront-Signature": "s"},
    })
    # Order preserved
    assert headers == {"Cookie": "CloudFront-Policy=p; CloudFront-Signature=s"}


def test_credential_to_headers_unknown_raises():
    with pytest.raises(CredentialMintError, match="unknown credential_type"):
        _credential_to_headers({"credential_type": "bogus"})


@pytest.mark.asyncio
async def test_private_dataset_with_role_mints_credentials():
    dataset = MagicMock(slug="brca", name="BRCA", description="",
                         is_public=False, required_roles=["researcher"],
                         chat_enabled=True, path="brca.zarr")
    datasource = MagicMock(base_url="https://example.com", type="HTTP_TOKEN",
                            credential_ref="brca_key")
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=["researcher"])
    llm = FakeLLMClient(scripts=[])

    fake_anndata = MagicMock()
    fake_anndata.n_obs = 10
    fake_anndata.n_vars = 20
    fake_anndata.obsm_keys = []
    fake_anndata.obs_columns = []

    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS, \
         patch("cell_explorer_api.services.chat_session.mint_credentials") as mock_mint:
        mock_mint.return_value = {
            "credential_type": "bearer_token",
            "url": "https://example.com/brca.zarr",
            "token": "xyz",
        }
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())

        agent = await make_chat_agent(
            user=user, dataset_slug="brca", db=db, settings=settings, llm=llm
        )

        mock_mint.assert_called_once_with(datasource, "brca.zarr")
        _, kwargs = MockZS.open.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer xyz"}
        assert agent is not None


@pytest.mark.asyncio
async def test_credential_mint_error_propagates_as_chat_session_error():
    dataset = MagicMock(slug="brca", is_public=False,
                         required_roles=["researcher"], chat_enabled=True, path="brca.zarr")
    datasource = MagicMock()
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=["researcher"])
    llm = FakeLLMClient(scripts=[])

    with patch("cell_explorer_api.services.chat_session.mint_credentials") as mock_mint:
        mock_mint.side_effect = CredentialError("private key missing")

        with pytest.raises(CredentialMintError, match="private key missing"):
            await make_chat_agent(
                user=user, dataset_slug="brca", db=db, settings=settings, llm=llm
            )


@pytest.mark.asyncio
async def test_chat_disabled_dataset_raises():
    dataset = MagicMock(
        slug="locked", is_public=True, required_roles=[],
        chat_enabled=False, path="locked.zarr",
    )
    datasource = MagicMock(base_url="https://example.com")
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=[])
    llm = FakeLLMClient(scripts=[])

    with pytest.raises(ChatDisabledError, match="locked"):
        await make_chat_agent(
            user=user, dataset_slug="locked", db=db, settings=settings, llm=llm
        )


@pytest.mark.asyncio
async def test_chat_enabled_dataset_passes_gate():
    """chat_enabled=True must let the call proceed past the new gate."""
    dataset = MagicMock(
        slug="open", is_public=True, required_roles=[],
        chat_enabled=True, path="open.zarr",
    )
    datasource = MagicMock(base_url="https://example.com")
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=[])
    llm = FakeLLMClient(scripts=[])

    fake_anndata = MagicMock()
    fake_anndata.n_obs = 10
    fake_anndata.n_vars = 20
    fake_anndata.obsm_keys = []
    fake_anndata.obs_columns = []
    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS:
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())
        agent = await make_chat_agent(
            user=user, dataset_slug="open", db=db, settings=settings, llm=llm
        )
        assert agent is not None


# ---------- Cache tests (issue #101) ----------

from datetime import datetime, timedelta, timezone


def _public_dataset(slug: str = "pbmc3k", *, updated_at: datetime | None = None) -> MagicMock:
    """Build a public-dataset mock with a settable updated_at."""
    ds = MagicMock(
        slug=slug,
        name=slug,
        description="",
        is_public=True,
        required_roles=[],
        chat_enabled=True,
        path=f"{slug}.zarr",
    )
    ds.updated_at = updated_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ds


@pytest.fixture(autouse=True)
def _clear_dataset_ctx_cache():
    """Each cache test starts with a clean cache."""
    from cell_explorer_api.services.chat_session import _dataset_ctx_cache
    _dataset_ctx_cache.clear()
    yield
    _dataset_ctx_cache.clear()


@pytest.mark.asyncio
async def test_dataset_ctx_cache_hits_on_second_call_with_same_slug_and_updated_at():
    """Two make_chat_agent calls with the same Dataset row build dataset_ctx
    only once — the second call hits the cache."""
    dataset = _public_dataset()
    datasource = MagicMock(base_url="https://example.com", type="HTTP_TOKEN", credential_ref=None)

    fake_anndata = MagicMock(n_obs=10, n_vars=20, obsm_keys=[], obs_columns=[])

    async def _db_factory():
        # Fresh db session per call (the test reuses the same dataset/datasource).
        return await _mk_db_session(_make_db_row(dataset, datasource))

    call_count = 0
    real_build = None

    async def _spy_build(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await real_build(*args, **kwargs)

    from cell_explorer_agent import build_dataset_context as _real
    real_build = _real

    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS, \
         patch("cell_explorer_api.services.chat_session.build_dataset_context", _spy_build):
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())

        user = _FakeUser(roles=[])
        llm = FakeLLMClient(scripts=[])
        settings = MagicMock()

        await make_chat_agent(user=user, dataset_slug="pbmc3k", db=await _db_factory(), settings=settings, llm=llm)
        await make_chat_agent(user=user, dataset_slug="pbmc3k", db=await _db_factory(), settings=settings, llm=llm)

    assert call_count == 1, f"build_dataset_context called {call_count}x; expected 1 (cache hit)"


@pytest.mark.asyncio
async def test_dataset_ctx_cache_invalidates_when_updated_at_changes():
    """After an admin PUT bumps the Dataset row's updated_at, the cache key
    changes — the next make_chat_agent call rebuilds dataset_ctx."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=1)
    dataset = _public_dataset(updated_at=t0)
    datasource = MagicMock(base_url="https://example.com", type="HTTP_TOKEN", credential_ref=None)

    fake_anndata = MagicMock(n_obs=10, n_vars=20, obsm_keys=[], obs_columns=[])

    call_count = 0
    from cell_explorer_agent import build_dataset_context as _real

    async def _spy_build(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await _real(*args, **kwargs)

    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS, \
         patch("cell_explorer_api.services.chat_session.build_dataset_context", _spy_build):
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())

        user = _FakeUser(roles=[])
        llm = FakeLLMClient(scripts=[])
        settings = MagicMock()

        # First call: cache miss, build_dataset_context runs.
        db1 = await _mk_db_session(_make_db_row(dataset, datasource))
        await make_chat_agent(user=user, dataset_slug="pbmc3k", db=db1, settings=settings, llm=llm)
        assert call_count == 1

        # Simulate an admin PUT bumping updated_at.
        dataset.updated_at = t1

        # Next call: cache key is different, build_dataset_context runs again.
        db2 = await _mk_db_session(_make_db_row(dataset, datasource))
        await make_chat_agent(user=user, dataset_slug="pbmc3k", db=db2, settings=settings, llm=llm)
        assert call_count == 2, f"build_dataset_context called {call_count}x; expected 2 after updated_at bump"


@pytest.mark.asyncio
async def test_dataset_ctx_cache_is_independent_per_slug():
    """Two datasets with different slugs cache independently."""
    ds_a = _public_dataset(slug="ds-a")
    ds_b = _public_dataset(slug="ds-b")
    datasource = MagicMock(base_url="https://example.com", type="HTTP_TOKEN", credential_ref=None)

    fake_anndata = MagicMock(n_obs=10, n_vars=20, obsm_keys=[], obs_columns=[])

    call_count = 0
    from cell_explorer_agent import build_dataset_context as _real

    async def _spy_build(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await _real(*args, **kwargs)

    with patch("cell_explorer_api.services.chat_session.ZarrStore") as MockZS, \
         patch("cell_explorer_api.services.chat_session.AnnDataStore") as MockADS, \
         patch("cell_explorer_api.services.chat_session.StrataStore") as MockSS, \
         patch("cell_explorer_api.services.chat_session.build_dataset_context", _spy_build):
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)
        MockSS.open = AsyncMock(return_value=MagicMock())

        user = _FakeUser(roles=[])
        llm = FakeLLMClient(scripts=[])
        settings = MagicMock()

        # ds-a cold: 1 build
        db_a = await _mk_db_session(_make_db_row(ds_a, datasource))
        await make_chat_agent(user=user, dataset_slug="ds-a", db=db_a, settings=settings, llm=llm)
        # ds-b cold: separate cache key, 1 more build
        db_b = await _mk_db_session(_make_db_row(ds_b, datasource))
        await make_chat_agent(user=user, dataset_slug="ds-b", db=db_b, settings=settings, llm=llm)
        # ds-a warm: cache hit
        db_a2 = await _mk_db_session(_make_db_row(ds_a, datasource))
        await make_chat_agent(user=user, dataset_slug="ds-a", db=db_a2, settings=settings, llm=llm)

    assert call_count == 2, f"build_dataset_context called {call_count}x; expected 2 (one per slug, ds-a's second is cached)"
