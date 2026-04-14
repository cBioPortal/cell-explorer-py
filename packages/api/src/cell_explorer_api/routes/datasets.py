"""Dataset discovery endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import Dataset, Datasource

router = APIRouter(tags=["datasets"])


class DatasetResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    is_public: bool
    url: str | None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]


def _user_can_access(dataset: Dataset, user: User | None) -> bool:
    """Check if user can access a dataset."""
    if dataset.is_public:
        return True
    if user is None:
        return False
    if not dataset.required_roles:
        return False
    return bool(set(user.roles) & set(dataset.required_roles))


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
        if _user_can_access(dataset, user):
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
    if not _user_can_access(dataset, user):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _dataset_to_response(dataset)
