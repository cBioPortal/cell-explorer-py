from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cell_explorer_agent import FakeLLMClient, Script
from cell_explorer_agent.config import AgentConfig

from cell_explorer_api.services.chat_session import (
    AccessDeniedError,
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
                         is_public=True, required_roles=[], path="pbmc3k.zarr")
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
         patch("cell_explorer_api.services.chat_session.mint_credentials") as mock_mint:
        MockZS.open = AsyncMock(return_value=MagicMock())
        MockADS.open = AsyncMock(return_value=fake_anndata)

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
                         required_roles=["researcher"])
    datasource = MagicMock()
    db = await _mk_db_session(_make_db_row(dataset, datasource))
    settings = MagicMock()
    user = _FakeUser(roles=["guest"])
    llm = FakeLLMClient(scripts=[])

    with pytest.raises(AccessDeniedError, match="researcher"):
        await make_chat_agent(
            user=user, dataset_slug="brca", db=db, settings=settings, llm=llm
        )
