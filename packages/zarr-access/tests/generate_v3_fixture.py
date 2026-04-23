"""Convert pbmc3k.zarr (v2) to pbmc3k_v3.zarr (v3) for testing."""

from pathlib import Path
import shutil
import anndata as ad

fixtures = Path(__file__).parent / "fixtures"
src = fixtures / "pbmc3k.zarr"
dst = fixtures / "pbmc3k_v3.zarr"

if dst.exists():
    shutil.rmtree(dst)

adata = ad.read_zarr(str(src))
ad.settings.zarr_write_format = 3
adata.write_zarr(str(dst))
print(f"Generated {dst}")
