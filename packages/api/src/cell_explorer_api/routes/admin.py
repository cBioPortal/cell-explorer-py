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
from cell_explorer_api.db.models import (
    Collection,
    Dataset,
    DatasetMetadata,
    Datasource,
    DatasourceType,
)
from cell_explorer_api.services.default_view import DefaultViewError, validate_default_view
from cell_explorer_api.services.metadata_harvest import harvest_and_store

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


# --- Collection schemas ---


class CollectionCreate(BaseModel):
    name: str
    slug: str
    description: str | None = None
    publication_url: str | None = None
    publication_citation: str | None = None


class CollectionUpdate(BaseModel):
    name: str | None = None
    slug: str | None = None
    description: str | None = None
    publication_url: str | None = None
    publication_citation: str | None = None


class CollectionAdminResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    publication_url: str | None
    publication_citation: str | None


class CollectionAdminListResponse(BaseModel):
    collections: list[CollectionAdminResponse]


# --- Dataset schemas ---


class DatasetCreate(BaseModel):
    datasource_id: str
    collection_id: str | None = None
    name: str
    slug: str
    path: str
    description: str | None = None
    is_public: bool = False
    required_roles: list[str] = []
    prompt_addendum: str | None = None
    default_view: dict | None = None
    chat_enabled: bool = False


class DatasetUpdate(BaseModel):
    name: str | None = None
    path: str | None = None
    description: str | None = None
    is_public: bool | None = None
    required_roles: list[str] | None = None
    prompt_addendum: str | None = None
    default_view: dict | None = None
    chat_enabled: bool | None = None
    collection_id: str | None = None


class DatasetMetadataAdminResponse(BaseModel):
    """Harvested store metadata, including harvest bookkeeping.

    Admin-only: `error` can contain the datasource's internal_base_url.
    """

    n_obs: int | None
    n_vars: int | None
    zarr_version: int | None
    obsm_keys: list[str]
    obs_columns: list[str]
    var_columns: list[str]
    layers: list[str]
    x_dtype: str | None
    x_encoding: str | None
    fetched_at: datetime | None
    last_attempt_at: datetime
    status: str
    error: str | None


class DatasetAdminResponse(BaseModel):
    id: str
    datasource_id: str
    collection_id: str | None
    name: str
    slug: str
    path: str
    description: str | None
    is_public: bool
    required_roles: list[str]
    prompt_addendum: str | None
    default_view: dict | None
    chat_enabled: bool
    metadata: DatasetMetadataAdminResponse | None = None


class DatasetAdminListResponse(BaseModel):
    datasets: list[DatasetAdminResponse]


def _dataset_to_admin_response(
    dataset: Dataset, metadata: DatasetMetadata | None = None
) -> DatasetAdminResponse:
    return DatasetAdminResponse(
        id=str(dataset.id),
        datasource_id=str(dataset.datasource_id),
        collection_id=str(dataset.collection_id) if dataset.collection_id else None,
        name=dataset.name,
        slug=dataset.slug,
        path=dataset.path,
        description=dataset.description,
        is_public=dataset.is_public,
        required_roles=dataset.required_roles,
        prompt_addendum=dataset.prompt_addendum,
        default_view=dataset.default_view,
        chat_enabled=dataset.chat_enabled,
        metadata=(
            DatasetMetadataAdminResponse(
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
                last_attempt_at=metadata.last_attempt_at,
                status=metadata.status,
                error=metadata.error,
            )
            if metadata is not None
            else None
        ),
    )


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
    statement = select(Dataset, DatasetMetadata).outerjoin(
        DatasetMetadata, DatasetMetadata.dataset_id == Dataset.id
    )
    result = await db.exec(statement)
    return DatasetAdminListResponse(
        datasets=[_dataset_to_admin_response(d, m) for d, m in result.all()]
    )


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
    if data.get("collection_id") is not None:
        try:
            data["collection_id"] = uuid.UUID(data["collection_id"])
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid collection_id format")
    if data.get("default_view") is not None:
        try:
            data["default_view"] = validate_default_view(data["default_view"])
        except DefaultViewError as exc:
            raise HTTPException(status_code=422, detail=f"invalid default_view: {exc}")
    dataset = Dataset(**data)
    db.add(dataset)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Dataset with this slug already exists")
    await db.refresh(dataset)

    # Best-effort: a store that cannot be read must not fail the catalog write.
    dataset.datasource = (
        await db.exec(select(Datasource).where(Datasource.id == dataset.datasource_id))
    ).one()
    metadata = await harvest_and_store(db, dataset, dataset.datasource)
    await db.commit()
    # commit() expires every object in the session, dataset included — refresh
    # it too, or the response build below triggers an implicit lazy-load that
    # raises MissingGreenlet under the async driver.
    await db.refresh(dataset)
    await db.refresh(metadata)

    return _dataset_to_admin_response(dataset, metadata)


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
    if updates.get("collection_id") is not None:
        try:
            updates["collection_id"] = uuid.UUID(updates["collection_id"])
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid collection_id format")
    if updates.get("default_view") is not None:
        try:
            updates["default_view"] = validate_default_view(updates["default_view"])
        except DefaultViewError as exc:
            raise HTTPException(status_code=422, detail=f"invalid default_view: {exc}")
    previous_path = dataset.path
    previous_datasource_id = dataset.datasource_id
    for key, value in updates.items():
        setattr(dataset, key, value)
    dataset.updated_at = datetime.now(timezone.utc)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    store_moved = (
        dataset.path != previous_path or dataset.datasource_id != previous_datasource_id
    )
    datasource = (
        await db.exec(select(Datasource).where(Datasource.id == dataset.datasource_id))
    ).one()

    if store_moved:
        metadata = await harvest_and_store(db, dataset, datasource)
        await db.commit()
        # See the same refresh in create_dataset: commit() expires dataset too.
        await db.refresh(dataset)
        await db.refresh(metadata)
    else:
        metadata = (
            await db.exec(
                select(DatasetMetadata).where(DatasetMetadata.dataset_id == dataset.id)
            )
        ).first()

    return _dataset_to_admin_response(dataset, metadata)


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


# --- Collection routes ---


def _collection_to_response(collection: Collection) -> CollectionAdminResponse:
    return CollectionAdminResponse(
        id=str(collection.id),
        name=collection.name,
        slug=collection.slug,
        description=collection.description,
        publication_url=collection.publication_url,
        publication_citation=collection.publication_citation,
    )


async def _get_collection_or_404(slug: str, db: AsyncSession) -> Collection:
    result = await db.exec(select(Collection).where(Collection.slug == slug))
    collection = result.first()
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


@router.post("/collections", status_code=201)
async def create_collection(
    payload: CollectionCreate,
    db: AsyncSession = Depends(get_db),
) -> CollectionAdminResponse:
    collection = Collection(**payload.model_dump())
    db.add(collection)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Collection slug already exists")
    await db.refresh(collection)
    return _collection_to_response(collection)


@router.get("/collections")
async def list_collections_admin(
    db: AsyncSession = Depends(get_db),
) -> CollectionAdminListResponse:
    result = await db.exec(select(Collection))
    return CollectionAdminListResponse(
        collections=[_collection_to_response(c) for c in result.all()]
    )


@router.put("/collections/{slug}")
async def update_collection(
    slug: str,
    payload: CollectionUpdate,
    db: AsyncSession = Depends(get_db),
) -> CollectionAdminResponse:
    collection = await _get_collection_or_404(slug, db)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(collection, key, value)
    collection.updated_at = datetime.now(timezone.utc)
    db.add(collection)
    await db.commit()
    await db.refresh(collection)
    return _collection_to_response(collection)


@router.delete("/collections/{slug}", status_code=204)
async def delete_collection(
    slug: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    collection = await _get_collection_or_404(slug, db)
    # Datasets survive: collection_id is ondelete="SET NULL".
    await db.delete(collection)
    await db.commit()
    return Response(status_code=204)
