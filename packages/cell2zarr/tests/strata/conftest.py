"""Shared fixtures for strata tests."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_anndata_zarr(tmp_path: Path) -> Path:
    """5 cells × 3 genes, two categorical obs columns. Hand-computable ground truth.

    Shape:
        cell_type: [A, A, B, B, C]
        donor:     [d1, d2, d1, d2, d1]
        X (float16):
            cell 0: [1, 2, 0]
            cell 1: [3, 4, 0]
            cell 2: [5, 0, 1]
            cell 3: [0, 0, 2]
            cell 4: [1, 1, 1]

    Atomic strata (cell_type × donor): 5 unique tuples (A,d1), (A,d2), (B,d1), (B,d2), (C,d1).
    """
    X = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [5.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float16,
    )
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "A", "B", "B", "C"]),
            "donor": pd.Categorical(["d1", "d2", "d1", "d2", "d1"]),
        }
    )
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    path = tmp_path / "tiny.zarr"
    adata.write_zarr(path)
    return path


@pytest.fixture(scope="session")
def pbmc3k_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """PBMC3K processed dataset as a zarr store. Cached across the test session.

    Downloads via scanpy on first use. ~80 MB.
    """
    import scanpy as sc

    cache_dir = tmp_path_factory.mktemp("pbmc3k_cache")
    path = cache_dir / "pbmc3k.zarr"
    adata = sc.datasets.pbmc3k_processed()
    # Make sure there's a categorical obs column suitable for atomic axes
    if "louvain" not in adata.obs.columns:
        sc.tl.leiden(adata, key_added="louvain")
    if not isinstance(adata.obs["louvain"].dtype, pd.CategoricalDtype):
        adata.obs["louvain"] = pd.Categorical(adata.obs["louvain"].astype(str))
    # Downcast X to float16 to match cell2zarr convention
    adata.X = adata.X.astype(np.float16)
    adata.write_zarr(path)
    return path
