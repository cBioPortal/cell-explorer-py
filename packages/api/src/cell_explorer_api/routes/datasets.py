"""Dataset discovery endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import Collection, Dataset, DatasetMetadata, Datasource
from cell_explorer_api.services.access import user_can_access
from cell_explorer_api.services.credentials import CredentialError, mint_credentials

router = APIRouter(tags=["datasets"])


class DatasetCollectionRef(BaseModel):
    slug: str
    name: str


class DatasetMetadataResponse(BaseModel):
    """Harvested store facts. Present only once a harvest has succeeded.

    Deliberately excludes `status` and `error` — error strings can contain the
    datasource's internal_base_url, which must not reach anonymous callers.
    Those live on the admin response instead.
    """

    n_obs: int
    n_vars: int
    zarr_version: int
    obsm_keys: list[str]
    obs_columns: list[str]
    var_columns: list[str]
    layers: list[str]
    x_dtype: str | None
    x_encoding: str | None
    fetched_at: datetime


class DatasetResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    is_public: bool
    url: str | None
    chat_enabled: bool
    default_view: dict | None = None
    collection: DatasetCollectionRef | None = None
    metadata: DatasetMetadataResponse | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]


def _metadata_to_response(
    metadata: DatasetMetadata | None,
) -> DatasetMetadataResponse | None:
    """Expose metadata only once a harvest has succeeded.

    fetched_at is the marker: it is set only on success, so a row that exists
    solely to record a failure yields None rather than a body of nulls. All
    seven value columns (n_obs, n_vars, zarr_version, obsm_keys, obs_columns,
    var_columns, layers) are checked together because a real harvest
    (store_harvest_result) writes all of them in a single block on success —
    the response type promises all seven are real (the list columns are
    non-optional `list[str]`), so a row missing any of them is not a genuine
    success and must not be served as though it were: doing so would raise a
    ValidationError building the response, a 500 on this unauthenticated
    route.
    """
    if (
        metadata is None
        or metadata.fetched_at is None
        or metadata.n_obs is None
        or metadata.n_vars is None
        or metadata.zarr_version is None
        or metadata.obsm_keys is None
        or metadata.obs_columns is None
        or metadata.var_columns is None
        or metadata.layers is None
    ):
        return None
    return DatasetMetadataResponse(
        n_obs=metadata.n_obs,
        n_vars=metadata.n_vars,
        zarr_version=metadata.zarr_version,
        obsm_keys=metadata.obsm_keys,
        obs_columns=metadata.obs_columns,
        var_columns=metadata.var_columns,
        layers=metadata.layers,
        x_dtype=metadata.x_dtype,
        x_encoding=metadata.x_encoding,
        fetched_at=metadata.fetched_at,
    )


def _dataset_to_response(
    dataset: Dataset, metadata: DatasetMetadata | None = None
) -> DatasetResponse:
    """Convert a Dataset model to API response."""
    url = None
    if dataset.is_public and dataset.datasource:
        url = f"{dataset.datasource.base_url}/{dataset.path}"
    return DatasetResponse(
        slug=dataset.slug,
        name=dataset.name,
        description=dataset.description,
        is_public=dataset.is_public,
        url=url,
        chat_enabled=dataset.chat_enabled,
        default_view=dataset.default_view,
        collection=(
            DatasetCollectionRef(slug=dataset.collection.slug, name=dataset.collection.name)
            if dataset.collection is not None
            else None
        ),
        metadata=_metadata_to_response(metadata),
    )


@router.get("/datasets")
async def list_datasets(
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    """List datasets the caller can access."""
    statement = (
        select(Dataset, Datasource, Collection, DatasetMetadata)
        .join(Datasource)
        .outerjoin(Collection)
        .outerjoin(DatasetMetadata, DatasetMetadata.dataset_id == Dataset.id)
    )
    result = await db.exec(statement)
    datasets = []
    for dataset, datasource, collection, metadata in result.all():
        dataset.datasource = datasource
        dataset.collection = collection
        if user_can_access(dataset, user=user):
            datasets.append(_dataset_to_response(dataset, metadata))
    return DatasetListResponse(datasets=datasets)


@router.get("/datasets/{slug}")
async def get_dataset(
    slug: str,
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Get a single dataset by slug."""
    statement = (
        select(Dataset, Datasource, Collection, DatasetMetadata)
        .join(Datasource)
        .outerjoin(Collection)
        .outerjoin(DatasetMetadata, DatasetMetadata.dataset_id == Dataset.id)
        .where(Dataset.slug == slug)
    )
    result = await db.exec(statement)
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset, datasource, collection, metadata = row
    dataset.datasource = datasource
    dataset.collection = collection
    if not user_can_access(dataset, user=user):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_to_response(dataset, metadata)


@router.post("/datasets/{slug}/access")
async def access_dataset(
    slug: str,
    request: Request,
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get access credentials for a dataset."""
    statement = select(Dataset, Datasource).join(Datasource).where(Dataset.slug == slug)
    result = await db.exec(statement)
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset, datasource = row
    dataset.datasource = datasource

    if dataset.is_public:
        return {
            "url": f"{datasource.base_url}/{dataset.path}",
            "credential_type": "public",
        }

    # Private dataset — require auth
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not set(user.roles) & set(dataset.required_roles):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return mint_credentials(datasource, dataset.path)
    except CredentialError as e:
        raise HTTPException(status_code=503, detail=str(e))
