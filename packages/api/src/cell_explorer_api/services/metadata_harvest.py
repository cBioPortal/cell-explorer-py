"""Fetch zarr store metadata for a dataset and persist it.

Never raises. Every failure — unreachable store, un-mintable credentials, a
store that is not AnnData — becomes a recorded error so the catalog write that
triggered the harvest still succeeds.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from zarr_access import ZarrStore

from cell_explorer_api.db.models import Dataset, DatasetMetadata, Datasource
from cell_explorer_api.services.credentials import credential_to_headers, mint_credentials
from cell_explorer_api.services.store_metadata import StoreMetadata, extract_store_metadata

logger = logging.getLogger(__name__)

# Truncated so a verbose upstream traceback cannot bloat the row.
MAX_ERROR_LENGTH = 500


def _utcnow() -> datetime:
    """UTC now, timezone-naive to match what SQLite returns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class HarvestOutcome:
    status: str  # "ok" | "error"
    metadata: StoreMetadata | None = None
    error: str | None = None


async def harvest_dataset_metadata(
    dataset: Dataset, datasource: Datasource
) -> HarvestOutcome:
    """Fetch store metadata for one dataset. Never raises."""
    # fetch_base_url, not base_url: the API and the browser have different
    # network views of a datasource (docker-compose maps zarr-server to a host
    # port the API container cannot reach). The credential token is signed for
    # the dataset path, not the URL, so it works either way.
    url = f"{datasource.fetch_base_url}/{dataset.path}"

    if dataset.is_public:
        headers: dict[str, str] = {}
    else:
        try:
            headers = credential_to_headers(mint_credentials(datasource, dataset.path))
        except Exception as exc:
            # Broad on purpose: mint_credentials/credential_to_headers can raise
            # more than CredentialError (e.g. OSError reading a corrupt/unreadable
            # HTTP_TOKEN private key file, jwt.exceptions.InvalidKeyError for bad
            # PEM contents). Every un-mintable-credential failure must become a
            # recorded error, never propagate.
            message = f"{type(exc).__name__}: {exc}"
            return HarvestOutcome(status="error", error=message[:MAX_ERROR_LENGTH])

    try:
        zarr_store = await ZarrStore.open(url, headers=headers)
        metadata = await extract_store_metadata(zarr_store)
    except Exception as exc:
        logger.warning("metadata harvest failed for dataset %s: %s", dataset.slug, exc)
        message = f"{type(exc).__name__}: {exc}"
        return HarvestOutcome(status="error", error=message[:MAX_ERROR_LENGTH])

    return HarvestOutcome(status="ok", metadata=metadata)


async def store_harvest_result(
    db: AsyncSession, dataset_id: uuid.UUID, outcome: HarvestOutcome
) -> DatasetMetadata:
    """Upsert the outcome.

    A success overwrites everything and clears the error. A failure updates only
    the bookkeeping columns, so a transient outage does not blank counts already
    being served.
    """
    result = await db.exec(
        select(DatasetMetadata).where(DatasetMetadata.dataset_id == dataset_id)
    )
    row = result.first()
    if row is None:
        row = DatasetMetadata(dataset_id=dataset_id)

    now = _utcnow()
    row.last_attempt_at = now
    row.status = outcome.status

    if outcome.status == "ok" and outcome.metadata is not None:
        md = outcome.metadata
        row.n_obs = md.n_obs
        row.n_vars = md.n_vars
        row.zarr_version = md.zarr_version
        row.obsm_keys = list(md.obsm_keys)
        row.obs_columns = list(md.obs_columns)
        row.var_columns = list(md.var_columns)
        row.layers = list(md.layers)
        row.x_dtype = md.x_dtype
        row.x_encoding = md.x_encoding
        row.fetched_at = now
        row.error = None
    else:
        row.error = outcome.error

    db.add(row)
    return row


async def harvest_and_store(
    db: AsyncSession, dataset: Dataset, datasource: Datasource
) -> DatasetMetadata:
    """Harvest and persist in one call. Does not commit."""
    outcome = await harvest_dataset_metadata(dataset, datasource)
    return await store_harvest_result(db, dataset.id, outcome)
