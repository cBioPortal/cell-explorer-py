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
        # Public datasource (no credentials needed)
        public_cdn = Datasource(
            name="Public CDN",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://public-data.cbioportal.org",
        )

        # Private datasource (requires credential_ref env vars)
        private_cdn = Datasource(
            name="MSK Internal",
            type=DatasourceType.HTTP_TOKEN,
            base_url="https://internal.msk.org/zarr",
            credential_ref="MSK_INTERNAL",
        )

        session.add(public_cdn)
        session.add(private_cdn)
        await session.flush()

        datasets = [
            Dataset(
                datasource_id=public_cdn.id,
                name="BRCA Tumor Atlas (Demo)",
                slug="brca-demo",
                path="brca-tumor-atlas/v1.zarr",
                description="Public demo breast cancer single-cell atlas",
                is_public=True,
            ),
            Dataset(
                datasource_id=private_cdn.id,
                name="PDX Models Internal",
                slug="pdx-internal",
                path="pdx-models/latest.zarr",
                description="Internal PDX model single-cell data",
                is_public=False,
                required_roles=["lab-pdx"],
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
