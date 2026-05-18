"""In-memory fake implementing StrataAccess for tests.

Deterministic data designed to align with FakeZarrAccess.default() so the
two fakes can be used together for integration-style tool tests.
"""

from dataclasses import dataclass, field

import numpy as np

from cell_explorer_agent.tools.strata_protocol import CoarseStrataResult


@dataclass
class FakeStrataAccess:
    """A tiny strata fixture mirroring FakeZarrAccess.default() shape.

    Default: 1 coarse table on `cell_type` with 3 strata (T cell, B cell,
    Monocyte) × 50 genes. Aggregates are hand-tuned to match what summing
    FakeZarrAccess.default()'s expression matrix grouped by cell_type would
    produce — but deterministic, not actually computed from the expression
    arrays. Tests that need numerical equivalence with X-scans should use
    the integration test (Task 7) against the real strata-tiny.zarr fixture.
    """

    coarse_tables: dict[str, CoarseStrataResult] = field(default_factory=dict)

    @classmethod
    def default(cls, var_names: list[str] | None = None) -> "FakeStrataAccess":
        n_strata = 3
        n_genes = len(var_names) if var_names is not None else 50
        stratum_keys = np.array([["T cell"], ["B cell"], ["Monocyte"]])
        # cell_type cardinalities: arbitrary-but-fixed split summing to 100
        n_cells = np.array([34, 33, 33], dtype=np.int32)
        # Deterministic sums: each stratum gets distinct per-gene patterns
        sum_x = np.zeros((n_strata, n_genes), dtype=np.float32)
        # T-cell stratum: gene 0 elevated so compare_groups can rank it
        sum_x[0, 0] = 200.0   # T cell gene 0
        sum_x[1, 0] = 20.0    # B cell gene 0
        sum_x[2, 0] = 20.0    # Monocyte gene 0
        if n_genes > 2:
            # gene 2: elevated in B cells
            sum_x[0, 2] = 10.0
            sum_x[1, 2] = 150.0
            sum_x[2, 2] = 10.0
        # Background expression for the other genes
        for j in range(n_genes):
            if j in (0, 2):
                continue
            sum_x[:, j] = [50.0, 50.0, 50.0]
        sum_xx = sum_x * sum_x  # rough placeholder
        nnz = np.full((n_strata, n_genes), 20, dtype=np.int32)
        return cls(coarse_tables={
            "cell_type": CoarseStrataResult(
                axes=["cell_type"],
                stratum_keys=stratum_keys,
                sum_x=sum_x,
                sum_xx=sum_xx,
                nnz=nnz,
                n_cells=n_cells,
            ),
        })

    @classmethod
    def with_axis(cls, axis: str, var_names: list[str] | None = None) -> "FakeStrataAccess":
        """Return a strata with one coarse table whose single axis is `axis`.

        The stratum_keys use generic labels (donor_1, donor_2, …) since
        callers of this factory only test routing (no group is fetched).
        """
        n_strata = 3
        n_genes = len(var_names) if var_names is not None else 50
        stratum_keys = np.array([[f"{axis}_1"], [f"{axis}_2"], [f"{axis}_3"]])
        n_cells = np.array([34, 33, 33], dtype=np.int32)
        sum_x = np.full((n_strata, n_genes), 10.0, dtype=np.float32)
        sum_xx = sum_x * sum_x
        nnz = np.full((n_strata, n_genes), 5, dtype=np.int32)
        return cls(coarse_tables={
            axis: CoarseStrataResult(
                axes=[axis],
                stratum_keys=stratum_keys,
                sum_x=sum_x,
                sum_xx=sum_xx,
                nnz=nnz,
                n_cells=n_cells,
            ),
        })

    @classmethod
    def from_zarr_data(cls, fake_zarr) -> "FakeStrataAccess":
        """Build a coarse strata for 'cell_type' whose sums match fake_zarr.expression
        exactly. Used by cross-path equivalence tests.
        """
        col = fake_zarr.obs["cell_type"]
        cats = list(col.categories)
        codes = col.values

        n_strata = len(cats)
        n_genes = len(fake_zarr.var)
        # Use float64 so the stored sums exactly match what the X-scan path
        # accumulates in float64. float32 storage would introduce ~1e-7 drift.
        sum_x = np.zeros((n_strata, n_genes), dtype="float64")
        sum_xx = np.zeros((n_strata, n_genes), dtype="float64")
        nnz = np.zeros((n_strata, n_genes), dtype="int32")
        n_cells = np.zeros(n_strata, dtype="int32")

        for s_idx, _cat_name in enumerate(cats):
            mask = codes == s_idx
            n_cells[s_idx] = int(mask.sum())
            for g_idx, gene in enumerate(fake_zarr.var):
                expr = fake_zarr.expression[gene]
                sub = expr[mask].astype("float64", copy=False)
                sum_x[s_idx, g_idx] = float(sub.sum())
                sum_xx[s_idx, g_idx] = float((sub * sub).sum())
                nnz[s_idx, g_idx] = int((sub != 0).sum())

        stratum_keys = np.array([[c] for c in cats], dtype=object)
        return cls(coarse_tables={
            "cell_type": CoarseStrataResult(
                axes=["cell_type"],
                stratum_keys=stratum_keys,
                sum_x=sum_x,
                sum_xx=sum_xx,
                nnz=nnz,
                n_cells=n_cells,
            ),
        })

    def coarse_slugs(self) -> list[str]:
        return sorted(self.coarse_tables.keys())

    def coarse_axes(self, slug: str) -> list[str]:
        if slug not in self.coarse_tables:
            raise KeyError(slug)
        return list(self.coarse_tables[slug].axes)

    async def read_coarse(self, slug: str) -> CoarseStrataResult:
        if slug not in self.coarse_tables:
            raise KeyError(slug)
        return self.coarse_tables[slug]
