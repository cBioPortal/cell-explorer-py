"""Shared factory for constructing a ready-to-run ChatAgent.

Used by the CLI today; will be reused by the future /api/chat HTTP route
(Plan 2).
"""

from typing import Any

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
from cell_explorer_api.services.zarr_adapter import AnnDataZarrAccess

# Heavy imports deferred to call time (via module-level import that can be patched in tests)
from zarr_access import AnnDataStore, ZarrStore


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

    # 1. Look up dataset + datasource
    stmt = select(Dataset, Datasource).join(Datasource).where(Dataset.slug == dataset_slug)
    result = await db.exec(stmt)
    row = result.first()
    if row is None:
        raise DatasetNotFoundError(f"dataset {dataset_slug!r} not found")
    dataset, datasource = row
    dataset.datasource = datasource  # match the pattern in routes/datasets.py

    # 2. Access check
    if not user_can_access(dataset, user=user):
        roles = ", ".join(dataset.required_roles) if dataset.required_roles else "(none)"
        raise AccessDeniedError(
            f"insufficient permissions for {dataset_slug!r}; required roles: {roles}"
        )

    # 2.5. Chat-disabled gate (Layer 3) — short-circuit before zarr work.
    if not dataset.chat_enabled:
        raise ChatDisabledError(
            f"chat is not enabled for dataset {dataset_slug!r}"
        )

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
    except Exception as exc:
        raise ZarrUnreachableError(str(exc)) from exc

    adapter = AnnDataZarrAccess(anndata)

    # 5. Build DatasetContext + tool catalog
    ctx = await build_dataset_context(
        adapter,
        slug=dataset.slug,
        name=dataset.name,
        description=dataset.description or "",
    )
    catalog = build_v1_catalog(adapter, config=agent_config)

    # 6. Construct LLMClient if not provided
    if llm is None:
        llm = AnthropicLLMClient(transport=agent_config.llm_transport)

    # 7. Return the wired agent
    return ChatAgent(llm=llm, catalog=catalog, dataset_ctx=ctx, config=agent_config)
