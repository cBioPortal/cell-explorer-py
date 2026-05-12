"""Seed the database with sample datasources and datasets for development."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType


async def seed(database_url: str) -> None:
    """Create tables and insert sample data."""
    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        # Public datasource — dev server serving zarr stores without auth
        public_dev = Datasource(
            name="Local Dev Server",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://localhost:3005",
        )

        # Private datasource — zarr-auth-proxy requiring JWT bearer tokens
        zarr_proxy = Datasource(
            name="Zarr Auth Proxy",
            type=DatasourceType.HTTP_TOKEN,
            base_url="http://localhost:8002",
            credential_ref="LOCAL",
        )

        session.add(public_dev)
        session.add(zarr_proxy)
        await session.flush()

        datasets = [
            Dataset(
                datasource_id=public_dev.id,
                name="PBMC 3K (Public)",
                slug="pbmc3k-public",
                path="pbmc3k.zarr",
                description="Public PBMC 3K single-cell dataset",
                is_public=True,
                chat_enabled=True,
            ),
            Dataset(
                datasource_id=zarr_proxy.id,
                name="PBMC 3K (Private)",
                slug="pbmc3k-private",
                path="pbmc3k.zarr",
                description="Private PBMC 3K behind zarr-auth-proxy",
                is_public=False,
                required_roles=["test-role"],
                chat_enabled=True,
            ),
        ]
        for ds in datasets:
            session.add(ds)

        await session.commit()

    await engine.dispose()
    print(f"Seeded {len(datasets)} datasets across 2 datasources.")


def main():
    settings = Settings()
    asyncio.run(seed(settings.effective_database_url))


if __name__ == "__main__":
    main()
