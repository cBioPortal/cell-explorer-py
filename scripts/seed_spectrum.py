"""Dev-stack seed: register the MSK SPECTRUM TME 2022 public zarr dataset.

Run this after `make db-migrate` (or any time you want a real public dataset
visible to the CLI / API for local testing). Idempotent — re-running is safe.

Usage:
    make seed-spectrum
    # or directly
    uv run --project packages/api python scripts/seed_spectrum.py
"""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType


async def main() -> None:
    settings = Settings()
    engine = create_async_engine(settings.effective_database_url)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        # Look up the datasource by base_url; create if missing
        base_url = "https://cbioportal-public-imaging.assets.cbioportal.org"
        result = await session.exec(select(Datasource).where(Datasource.base_url == base_url))
        datasource = result.first()
        if datasource is None:
            datasource = Datasource(
                name="cBioPortal Public Imaging",
                type=DatasourceType.S3_CLOUDFRONT,
                base_url=base_url,
            )
            session.add(datasource)
            await session.flush()

        # Look up the dataset by slug; create if missing
        slug = "spectrum"
        result = await session.exec(select(Dataset).where(Dataset.slug == slug))
        dataset = result.first()
        if dataset is None:
            dataset = Dataset(
                datasource_id=datasource.id,
                slug=slug,
                name="MSK SPECTRUM TME 2022",
                path="msk_spectrum_tme_2022/zarr/spectrum_all_cells-f16-zstd-c1s30-v3.zarr",
                description="MSK SPECTRUM tumor microenvironment single-cell atlas (2022)",
                is_public=True,
            )
            session.add(dataset)
            await session.commit()
            print(f"Created dataset slug={slug}")
        else:
            print(f"Dataset slug={slug} already exists; no change")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
