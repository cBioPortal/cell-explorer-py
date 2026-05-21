"""Shared factory for constructing a ready-to-run ChatAgent.

Used by the CLI today; will be reused by the future /api/chat HTTP route
(Plan 2).
"""

from datetime import datetime
from typing import Any

from cell_explorer_agent.prompt import DatasetContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_agent import ChatAgent, build_dataset_context
from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.llm.anthropic import AnthropicLLMClient
from cell_explorer_agent.llm.base import LLMClient
from cell_explorer_agent.tools import build_v1_catalog

from cell_explorer_api.auth.models import User
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource
from cell_explorer_api.services.access import user_can_access
from cell_explorer_api.services.credentials import CredentialError, mint_credentials
from cell_explorer_api.services.zarr_adapter import (
    AnnDataZarrAccess,
    StrataZarrAccess,
)

# Heavy imports deferred to call time (via module-level import that can be patched in tests)
from zarr_access import AnnDataStore, StrataStore, ZarrStore


# In-process cache of DatasetContext keyed by (slug, updated_at). The Dataset
# row's updated_at bumps on every admin PUT (see routes/admin.py), so this
# self-invalidates without an explicit hook: admin edits change the key,
# subsequent requests miss the cache and rebuild. Process restart clears the
# cache (uvicorn --reload covers dev). Stale entries accumulate on edits but
# the leak is bounded by edit frequency and dataset count.
#
# See issue #101.
_dataset_ctx_cache: dict[tuple[str, datetime], DatasetContext] = {}


async def _build_dataset_context_cached(
    adapter: AnnDataZarrAccess,
    *,
    dataset: Dataset,
) -> DatasetContext:
    """Cached wrapper around build_dataset_context.

    Cached by (slug, updated_at). build_dataset_context reads every obs
    column from the zarr store (categorical codes + categories arrays for
    each), which is the dominant cost of /context cold-start. Caching the
    DatasetContext means subsequent requests for the same Dataset row pay
    only the cost of opening the zarr store, not the per-column reads.
    """
    key = (dataset.slug, dataset.updated_at)
    if key not in _dataset_ctx_cache:
        _dataset_ctx_cache[key] = await build_dataset_context(
            adapter,
            slug=dataset.slug,
            name=dataset.name,
            description=dataset.description or "",
            prompt_addendum=dataset.prompt_addendum,
        )
    return _dataset_ctx_cache[key]


class ChatSessionError(Exception):
    """Base for errors raised during chat-agent construction."""


class DatasetNotFoundError(ChatSessionError):
    pass


class AccessDeniedError(ChatSessionError):
    pass


class CredentialMintError(ChatSessionError):
    pass


class ZarrUnreachableError(ChatSessionError):
    pass


class ChatDisabledError(ChatSessionError):
    """Raised when chat is not enabled for the requested dataset."""


def _credential_to_headers(credential: dict[str, Any]) -> dict[str, str]:
    """Translate mint_credentials output into HTTP headers for ZarrStore.open."""
    kind = credential.get("credential_type")
    if kind == "public":
        return {}
    if kind == "bearer_token":
        return {"Authorization": f"Bearer {credential['token']}"}
    if kind == "signed_cookies":
        cookies = credential.get("cookies") or {}
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        return {"Cookie": cookie_header}
    raise CredentialMintError(f"unknown credential_type {kind!r}")


async def assert_chat_access(
    *,
    user: User,
    dataset_slug: str,
    db: AsyncSession,
) -> Dataset:
    """Run the chat gating cascade WITHOUT building an agent.

    Performs dataset lookup, dataset-access check, and per-dataset chat
    opt-in check — the same first three checks `make_chat_agent` does —
    but stops short of minting credentials and opening the zarr store.

    Used by the cheap endpoints (/threads, /threads/{id}, DELETE) that
    need the same authorization but don't need a working LLM session.

    Returns the resolved Dataset (with datasource attached) for use as
    an FK target.
    """
    stmt = select(Dataset, Datasource).join(Datasource).where(Dataset.slug == dataset_slug)
    result = await db.exec(stmt)
    row = result.first()
    if row is None:
        raise DatasetNotFoundError(f"dataset {dataset_slug!r} not found")
    dataset, datasource = row
    dataset.datasource = datasource

    if not user_can_access(dataset, user=user):
        roles = ", ".join(dataset.required_roles) if dataset.required_roles else "(none)"
        raise AccessDeniedError(
            f"insufficient permissions for {dataset_slug!r}; required roles: {roles}"
        )

    if not dataset.chat_enabled:
        raise ChatDisabledError(
            f"chat is not enabled for dataset {dataset_slug!r}"
        )

    return dataset


async def make_chat_agent(
    *,
    user: User,
    dataset_slug: str,
    db: AsyncSession,
    settings: Settings,
    agent_config: AgentConfig | None = None,
    llm: LLMClient | None = None,
) -> ChatAgent:
    """Build a ChatAgent with real auth + credentials for the given dataset."""
    agent_config = agent_config or AgentConfig()

    # 1, 2, 2.5: dataset lookup + access check + chat-enabled gate
    dataset = await assert_chat_access(user=user, dataset_slug=dataset_slug, db=db)
    datasource = dataset.datasource
    assert datasource is not None  # assert_chat_access attaches it

    # 3. Mint credentials (private only) and compute URL + headers.
    # Server-side fetches must use fetch_base_url (internal_base_url || base_url)
    # because the API and the client may have different network views of the
    # datasource (docker-compose: container-network hostname vs host-port URL).
    # The credential token (if any) is signed for the dataset path, not the URL,
    # so the same token works regardless of which URL fetches it.
    if dataset.is_public:
        url = f"{datasource.fetch_base_url}/{dataset.path}"
        headers: dict[str, str] = {}
    else:
        try:
            credential = mint_credentials(datasource, dataset.path)
        except CredentialError as exc:
            raise CredentialMintError(str(exc)) from exc
        url = f"{datasource.fetch_base_url}/{dataset.path}"
        headers = _credential_to_headers(credential)

    # 4. Open the zarr stack
    try:
        zarr_store = await ZarrStore.open(url, headers=headers)
        anndata = await AnnDataStore.open(zarr_store)
        strata_store = await StrataStore.open(zarr_store)
    except Exception as exc:
        raise ZarrUnreachableError(str(exc)) from exc

    adapter = AnnDataZarrAccess(anndata)
    strata_adapter = StrataZarrAccess(strata_store)

    # 5. Build DatasetContext + tool catalog (ctx is cached per (slug, updated_at))
    ctx = await _build_dataset_context_cached(adapter, dataset=dataset)
    # Only register the strata-powered tool if the dataset has coarse tables.
    # Empty -> pass None so the LLM only sees the X-scan compare_groups.
    strata_for_catalog = strata_adapter if strata_adapter.coarse_slugs() else None
    catalog = build_v1_catalog(adapter, config=agent_config, strata=strata_for_catalog)

    # 6. Construct LLMClient if not provided
    if llm is None:
        llm = AnthropicLLMClient(transport=agent_config.llm_transport)

    # 7. Return the wired agent
    return ChatAgent(llm=llm, catalog=catalog, dataset_ctx=ctx, config=agent_config)
