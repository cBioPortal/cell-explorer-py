"""Dataset discovery endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import Dataset, Datasource
from cell_explorer_api.services.access import user_can_access
from cell_explorer_api.services.credentials import CredentialError, mint_credentials

router = APIRouter(tags=["datasets"])


class DatasetResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    is_public: bool
    url: str | None
    chat_enabled: bool


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]


def _dataset_to_response(dataset: Dataset) -> DatasetResponse:
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
    )


@router.get("/datasets")
async def list_datasets(
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    """List datasets the caller can access."""
    statement = select(Dataset, Datasource).join(Datasource)
    result = await db.exec(statement)
    datasets = []
    for dataset, datasource in result.all():
        dataset.datasource = datasource
        if user_can_access(dataset, user=user):
            datasets.append(_dataset_to_response(dataset))
    return DatasetListResponse(datasets=datasets)


@router.get("/datasets/{slug}")
async def get_dataset(
    slug: str,
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Get a single dataset by slug."""
    statement = select(Dataset, Datasource).join(Datasource).where(Dataset.slug == slug)
    result = await db.exec(statement)
    row = result.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset, datasource = row
    dataset.datasource = datasource
    if not user_can_access(dataset, user=user):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_to_response(dataset)


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
