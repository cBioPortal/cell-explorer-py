"""Collection discovery endpoints.

A collection has no access fields of its own. It is visible exactly when the
caller can access at least one dataset in it, so `user_can_access` stays the
only access rule in the system.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from cell_explorer_api.auth.models import User
from cell_explorer_api.auth.optional import optional_auth
from cell_explorer_api.db import get_db
from cell_explorer_api.db.models import Collection, Dataset, DatasetMetadata, Datasource
from cell_explorer_api.routes.datasets import DatasetResponse, _dataset_to_response
from cell_explorer_api.services.access import user_can_access

router = APIRouter(tags=["collections"])


class CollectionSummary(BaseModel):
    slug: str
    name: str
    description: str | None
    publication_url: str | None
    publication_citation: str | None
    dataset_count: int


class CollectionListResponse(BaseModel):
    collections: list[CollectionSummary]


class CollectionDetail(CollectionSummary):
    datasets: list[DatasetResponse]


async def _accessible_by_collection(
    db: AsyncSession, user: User | None, *, collection_id: uuid.UUID | None = None
) -> dict[uuid.UUID, list[tuple[Dataset, DatasetMetadata | None]]]:
    """Group the caller's accessible datasets by collection id.

    One query for every dataset, then grouped in Python -- the same shape
    `list_datasets` uses. Never one query per collection.

    `collection_id` narrows the query to a single collection's datasets. The
    detail endpoint always passes it: without it, an existing-but-inaccessible
    collection would run the same full-table scan as an unknown slug takes
    zero of, making the two 404s distinguishable by timing even though their
    bodies are identical.

    Metadata is outer-joined in the same query, never fetched per dataset, so
    the collection detail route can render the same table as the catalogue.
    """
    statement = (
        select(Dataset, Datasource, Collection, DatasetMetadata)
        .join(Datasource)
        .outerjoin(Collection)
        .outerjoin(DatasetMetadata, DatasetMetadata.dataset_id == Dataset.id)
    )
    if collection_id is not None:
        statement = statement.where(Dataset.collection_id == collection_id)
    result = await db.exec(statement)
    grouped: dict[uuid.UUID, list[tuple[Dataset, DatasetMetadata | None]]] = {}
    for dataset, datasource, collection, metadata in result.all():
        if dataset.collection_id is None:
            continue
        dataset.datasource = datasource
        dataset.collection = collection
        if not user_can_access(dataset, user=user):
            continue
        grouped.setdefault(dataset.collection_id, []).append((dataset, metadata))
    return grouped


def _summary(collection: Collection, count: int) -> CollectionSummary:
    return CollectionSummary(
        slug=collection.slug,
        name=collection.name,
        description=collection.description,
        publication_url=collection.publication_url,
        publication_citation=collection.publication_citation,
        dataset_count=count,
    )


@router.get("/collections")
async def list_collections(
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> CollectionListResponse:
    """List collections containing at least one dataset the caller can access."""
    grouped = await _accessible_by_collection(db, user)
    result = await db.exec(select(Collection))
    summaries = [
        _summary(collection, len(grouped[collection.id]))
        for collection in result.all()
        if grouped.get(collection.id)
    ]
    summaries.sort(key=lambda c: c.name)
    return CollectionListResponse(collections=summaries)


@router.get("/collections/{slug}")
async def get_collection(
    slug: str,
    user: User | None = Depends(optional_auth),
    db: AsyncSession = Depends(get_db),
) -> CollectionDetail:
    """Get one collection and the datasets in it the caller can access."""
    result = await db.exec(select(Collection).where(Collection.slug == slug))
    collection = result.first()
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    grouped = await _accessible_by_collection(db, user, collection_id=collection.id)
    datasets = grouped.get(collection.id, [])
    if not datasets:
        # 404 rather than 403: a 403 would confirm this collection exists.
        raise HTTPException(status_code=404, detail="Collection not found")

    summary = _summary(collection, len(datasets))
    return CollectionDetail(
        **summary.model_dump(),
        datasets=[_dataset_to_response(dataset, metadata) for dataset, metadata in datasets],
    )
