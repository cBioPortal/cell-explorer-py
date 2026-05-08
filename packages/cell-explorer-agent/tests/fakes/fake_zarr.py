"""In-memory fake implementing ZarrAccess for tests."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cell_explorer_agent.tools.zarr_protocol import ObsColumn, ObsColumnSpec


@dataclass
class FakeZarrAccess:
    """A tiny AnnData-like fixture for tool tests.

    Default: 100 cells, 50 genes. `cell_type` has 3 categories.
    """

    n_obs: int = 100
    n_var: int = 50
    obs: dict[str, ObsColumn] = field(default_factory=dict)
    var: list[str] = field(default_factory=list)
    var_columns_data: list[str] = field(default_factory=lambda: ["feature_id", "gene_symbol"])
    obsm: list[str] = field(default_factory=lambda: ["X_umap", "X_pca"])
    expression: dict[str, np.ndarray] = field(default_factory=dict)
    _attrs: dict = field(
        default_factory=lambda: {"encoding-type": "anndata", "encoding-version": "0.1.0"}
    )

    @classmethod
    def default(cls) -> "FakeZarrAccess":
        rng = np.random.default_rng(0)
        n_obs, n_var = 100, 50
        obs_index = np.array([f"cell{i}" for i in range(n_obs)])
        cats = np.array(["T cell", "B cell", "Monocyte"])
        codes = rng.integers(0, 3, size=n_obs)
        n_counts = rng.normal(5000, 1500, size=n_obs).astype("float32")
        var = [f"gene{i}" for i in range(n_var)]
        # tag a couple of "marker" genes we can test queries against
        var[0] = "CD8A"
        var[1] = "CD4"
        var[2] = "MS4A1"
        expression = {
            g: rng.random(n_obs, dtype="float32") * 10 for g in var
        }
        # CD8A higher in T cells
        t_mask = codes == 0
        expression["CD8A"][t_mask] += 5
        return cls(
            n_obs=n_obs,
            n_var=n_var,
            obs={
                "cell_type": ObsColumn(
                    name="cell_type",
                    dtype="categorical",
                    values=codes,
                    categories=list(cats),
                    index=obs_index,
                ),
                "n_counts": ObsColumn(
                    name="n_counts",
                    dtype="numeric",
                    values=n_counts,
                    index=obs_index,
                ),
            },
            var=var,
            expression=expression,
        )

    async def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_var)

    async def attrs(self) -> dict:
        return self._attrs

    async def obs_columns(self) -> list[ObsColumnSpec]:
        return [
            ObsColumnSpec(
                name=c.name,
                dtype=c.dtype,
                cardinality=(len(c.categories) if c.categories else None),
                categories=list(c.categories) if c.categories else None,
            )
            for c in self.obs.values()
        ]

    async def obs_column(self, name: str) -> ObsColumn:
        if name not in self.obs:
            raise KeyError(name)
        return self.obs[name]

    async def var_names(self) -> list[str]:
        return list(self.var)

    async def var_columns(self) -> list[str]:
        return list(self.var_columns_data)

    async def obsm_keys(self) -> list[str]:
        return list(self.obsm)

    async def gene_index(self, gene: str) -> int:
        return self.var.index(gene)

    async def gene_column(self, gene: str) -> np.ndarray:
        if gene not in self.expression:
            raise KeyError(gene)
        return self.expression[gene]

    async def obs_mask(self, obs_col: str, value: str) -> np.ndarray:
        col = self.obs[obs_col]
        assert col.dtype == "categorical" and col.categories is not None
        code = col.categories.index(value)
        return col.values == code
