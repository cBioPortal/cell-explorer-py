"""Tests for database models."""

import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import SQLModel

from cell_explorer_api.db.models import Datasource, DatasourceType, Dataset


def test_datasource_create():
    ds = Datasource(
        name="MSK CloudFront",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://d1234.cloudfront.net",
        credential_ref="MSK_CLOUDFRONT",
    )
    assert ds.name == "MSK CloudFront"
    assert ds.type == DatasourceType.S3_CLOUDFRONT
    assert ds.base_url == "https://d1234.cloudfront.net"
    assert ds.credential_ref == "MSK_CLOUDFRONT"
    assert ds.id is not None
    assert isinstance(ds.created_at, datetime)


def test_datasource_credential_ref_nullable():
    ds = Datasource(
        name="Public CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://public.cdn.net",
    )
    assert ds.credential_ref is None


def test_dataset_create():
    datasource_id = uuid.uuid4()
    dataset = Dataset(
        datasource_id=datasource_id,
        name="BRCA Tumor Atlas",
        slug="brca-tumor-atlas",
        path="datasets/brca.zarr",
        is_public=True,
    )
    assert dataset.name == "BRCA Tumor Atlas"
    assert dataset.slug == "brca-tumor-atlas"
    assert dataset.path == "datasets/brca.zarr"
    assert dataset.is_public is True
    assert dataset.required_roles == []
    assert dataset.description is None


def test_dataset_with_roles():
    datasource_id = uuid.uuid4()
    dataset = Dataset(
        datasource_id=datasource_id,
        name="Private Dataset",
        slug="private-dataset",
        path="datasets/private.zarr",
        is_public=False,
        required_roles=["lab-smith", "project-alpha"],
    )
    assert dataset.required_roles == ["lab-smith", "project-alpha"]


def test_dataset_url_property():
    ds = Datasource(
        name="CDN",
        type=DatasourceType.S3_CLOUDFRONT,
        base_url="https://d1234.cloudfront.net",
    )
    dataset = Dataset(
        datasource_id=ds.id,
        name="Test",
        slug="test",
        path="datasets/test.zarr",
        is_public=True,
    )
    dataset.datasource = ds
    assert dataset.url == "https://d1234.cloudfront.net/datasets/test.zarr"


def test_datasource_type_enum():
    assert DatasourceType.S3_CLOUDFRONT == "s3_cloudfront"
    assert DatasourceType.HTTP_TOKEN == "http_token"
