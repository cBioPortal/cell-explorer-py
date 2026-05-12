"""Tests for admin CRUD endpoints."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Datasource, DatasourceType
from cell_explorer_api.main import create_app

API_KEY = "test-admin-key"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def app():
    settings = Settings(
        database_url="sqlite+aiosqlite://",
        admin_api_key=API_KEY,
    )
    return create_app(settings)


@pytest_asyncio.fixture()
async def ready_app(app):
    """App with tables created."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    app.state.db_engine = engine
    return app


@pytest_asyncio.fixture()
async def seeded_app(ready_app):
    """App with a datasource pre-seeded."""
    engine = ready_app.state.db_engine
    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="Test CDN",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://cdn.example.com",
            credential_ref="TEST_CDN",
        )
        session.add(ds)
        await session.commit()
        await session.refresh(ds)
        ready_app.state.test_datasource_id = str(ds.id)
    return ready_app


# --- Datasource CRUD ---

def test_create_datasource(ready_app):
    client = TestClient(ready_app)
    response = client.post(
        "/api/admin/datasources",
        json={
            "name": "New CDN",
            "type": "s3_cloudfront",
            "base_url": "https://new.cdn.net",
            "credential_ref": "NEW_CDN",
        },
        headers=AUTH_HEADER,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New CDN"
    assert data["type"] == "s3_cloudfront"
    assert "id" in data


def test_create_datasource_with_internal_base_url(ready_app):
    client = TestClient(ready_app)
    response = client.post(
        "/api/admin/datasources",
        json={
            "name": "Compose Split",
            "type": "http_token",
            "base_url": "http://localhost:8002",
            "internal_base_url": "http://zarr-server:8000",
        },
        headers=AUTH_HEADER,
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["base_url"] == "http://localhost:8002"
    assert data["internal_base_url"] == "http://zarr-server:8000"


def test_update_datasource_internal_base_url(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    response = client.put(
        f"/api/admin/datasources/{ds_id}",
        json={"internal_base_url": "http://zarr-server:8000"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200, response.text
    assert response.json()["internal_base_url"] == "http://zarr-server:8000"


def test_list_datasources(seeded_app):
    client = TestClient(seeded_app)
    response = client.get("/api/admin/datasources", headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert len(data["datasources"]) == 1
    assert data["datasources"][0]["name"] == "Test CDN"


def test_update_datasource(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    response = client.put(
        f"/api/admin/datasources/{ds_id}",
        json={"name": "Updated CDN"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated CDN"


def test_admin_requires_auth(ready_app):
    client = TestClient(ready_app)
    response = client.get("/api/admin/datasources")
    assert response.status_code == 403


# --- Dataset CRUD ---

def test_create_dataset(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    response = client.post(
        "/api/admin/datasets",
        json={
            "datasource_id": ds_id,
            "name": "Test Atlas",
            "slug": "test-atlas",
            "path": "datasets/test.zarr",
            "is_public": True,
        },
        headers=AUTH_HEADER,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["slug"] == "test-atlas"


def test_list_datasets_includes_private(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    client.post(
        "/api/admin/datasets",
        json={
            "datasource_id": ds_id,
            "name": "Private Atlas",
            "slug": "private-atlas",
            "path": "datasets/private.zarr",
            "is_public": False,
            "required_roles": ["lab-smith"],
        },
        headers=AUTH_HEADER,
    )
    response = client.get("/api/admin/datasets", headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert len(data["datasets"]) == 1
    assert data["datasets"][0]["slug"] == "private-atlas"
    assert data["datasets"][0]["required_roles"] == ["lab-smith"]


def test_create_dataset_duplicate_slug(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    payload = {
        "datasource_id": ds_id,
        "name": "Atlas",
        "slug": "same-slug",
        "path": "datasets/a.zarr",
        "is_public": True,
    }
    client.post("/api/admin/datasets", json=payload, headers=AUTH_HEADER)
    response = client.post("/api/admin/datasets", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 409


def test_update_dataset(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    # Create first
    client.post(
        "/api/admin/datasets",
        json={
            "datasource_id": ds_id,
            "name": "Original",
            "slug": "update-me",
            "path": "datasets/x.zarr",
            "is_public": False,
            "required_roles": ["lab-a"],
        },
        headers=AUTH_HEADER,
    )
    # Update
    response = client.put(
        "/api/admin/datasets/update-me",
        json={"name": "Updated", "required_roles": ["lab-a", "lab-b"]},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated"
    assert data["required_roles"] == ["lab-a", "lab-b"]


def test_delete_dataset(seeded_app):
    ds_id = seeded_app.state.test_datasource_id
    client = TestClient(seeded_app)
    client.post(
        "/api/admin/datasets",
        json={
            "datasource_id": ds_id,
            "name": "Delete Me",
            "slug": "delete-me",
            "path": "datasets/d.zarr",
            "is_public": True,
        },
        headers=AUTH_HEADER,
    )
    response = client.delete("/api/admin/datasets/delete-me", headers=AUTH_HEADER)
    assert response.status_code == 204
    # Verify it's gone
    response = client.get("/api/datasets/delete-me")
    assert response.status_code == 404
