"""Admin CRUD endpoints for datasources and datasets."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_api.auth.admin import require_admin
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# --- Datasource schemas ---


class DatasourceCreate(BaseModel):
    name: str
    type: DatasourceType
    base_url: str
    internal_base_url: str | None = None
    credential_ref: str | None = None


class DatasourceUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    internal_base_url: str | None = None
    credential_ref: str | None = None


class DatasourceResponse(BaseModel):
    id: str
    name: str
    type: DatasourceType
    base_url: str
    internal_base_url: str | None
    credential_ref: str | None


class DatasourceListResponse(BaseModel):
    datasources: list[DatasourceResponse]


# --- Dataset schemas ---


class DatasetCreate(BaseModel):
    datasource_id: str
    name: str
    slug: str
    path: str
    description: str | None = None
    is_public: bool = False
    required_roles: list[str] = []
    chat_enabled: bool = False


class DatasetUpdate(BaseModel):
    name: str | None = None
    path: str | None = None
    description: str | None = None
    is_public: bool | None = None
    required_roles: list[str] | None = None
    chat_enabled: bool | None = None


class DatasetAdminResponse(BaseModel):
    id: str
    datasource_id: str
    name: str
    slug: str
    path: str
    description: str | None
    is_public: bool
    required_roles: list[str]
    chat_enabled: bool


class DatasetAdminListResponse(BaseModel):
    datasets: list[DatasetAdminResponse]


# --- Datasource routes ---


@router.post("/datasources", status_code=201)
async def create_datasource(
    payload: DatasourceCreate,
    db: AsyncSession = Depends(get_db),
) -> DatasourceResponse:
    ds = Datasource(**payload.model_dump())
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return DatasourceResponse(
        id=str(ds.id),
        name=ds.name,
        type=ds.type,
        base_url=ds.base_url,
        internal_base_url=ds.internal_base_url,
        credential_ref=ds.credential_ref,
    )


@router.get("/datasources")
async def list_datasources(
    db: AsyncSession = Depends(get_db),
) -> DatasourceListResponse:
    result = await db.exec(select(Datasource))
    datasources = [
        DatasourceResponse(
            id=str(ds.id),
            name=ds.name,
            type=ds.type,
            base_url=ds.base_url,
            internal_base_url=ds.internal_base_url,
        credential_ref=ds.credential_ref,
        )
        for ds in result.all()
    ]
    return DatasourceListResponse(datasources=datasources)


@router.put("/datasources/{datasource_id}")
async def update_datasource(
    datasource_id: str,
    payload: DatasourceUpdate,
    db: AsyncSession = Depends(get_db),
) -> DatasourceResponse:
    try:
        ds_uuid = uuid.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Datasource not found")
    ds = await db.get(Datasource, ds_uuid)
    if ds is None:
        raise HTTPException(status_code=404, detail="Datasource not found")
    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(ds, key, value)
    ds.updated_at = datetime.now(timezone.utc)
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return DatasourceResponse(
        id=str(ds.id),
        name=ds.name,
        type=ds.type,
        base_url=ds.base_url,
        internal_base_url=ds.internal_base_url,
        credential_ref=ds.credential_ref,
    )


# --- Dataset routes ---


@router.get("/datasets")
async def list_datasets_admin(
    db: AsyncSession = Depends(get_db),
) -> DatasetAdminListResponse:
    result = await db.exec(select(Dataset))
    datasets = [
        DatasetAdminResponse(
            id=str(dataset.id),
            datasource_id=str(dataset.datasource_id),
            name=dataset.name,
            slug=dataset.slug,
            path=dataset.path,
            description=dataset.description,
            is_public=dataset.is_public,
            required_roles=dataset.required_roles,
            chat_enabled=dataset.chat_enabled,
        )
        for dataset in result.all()
    ]
    return DatasetAdminListResponse(datasets=datasets)


@router.post("/datasets", status_code=201)
async def create_dataset(
    payload: DatasetCreate,
    db: AsyncSession = Depends(get_db),
) -> DatasetAdminResponse:
    data = payload.model_dump()
    try:
        data["datasource_id"] = uuid.UUID(data["datasource_id"])
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid datasource_id format")
    dataset = Dataset(**data)
    db.add(dataset)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Dataset with this slug already exists")
    await db.refresh(dataset)
    return DatasetAdminResponse(
        id=str(dataset.id),
        datasource_id=str(dataset.datasource_id),
        name=dataset.name,
        slug=dataset.slug,
        path=dataset.path,
        description=dataset.description,
        is_public=dataset.is_public,
        required_roles=dataset.required_roles,
        chat_enabled=dataset.chat_enabled,
    )


@router.put("/datasets/{slug}")
async def update_dataset(
    slug: str,
    payload: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
) -> DatasetAdminResponse:
    result = await db.exec(select(Dataset).where(Dataset.slug == slug))
    dataset = result.first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(dataset, key, value)
    dataset.updated_at = datetime.now(timezone.utc)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return DatasetAdminResponse(
        id=str(dataset.id),
        datasource_id=str(dataset.datasource_id),
        name=dataset.name,
        slug=dataset.slug,
        path=dataset.path,
        description=dataset.description,
        is_public=dataset.is_public,
        required_roles=dataset.required_roles,
        chat_enabled=dataset.chat_enabled,
    )


@router.delete("/datasets/{slug}", status_code=204)
async def delete_dataset(
    slug: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    result = await db.exec(select(Dataset).where(Dataset.slug == slug))
    dataset = result.first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await db.delete(dataset)
    await db.commit()
    return Response(status_code=204)
