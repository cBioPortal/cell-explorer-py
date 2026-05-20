"""In-memory fake implementing StrataAccess for tests.

Deterministic data designed to align with FakeZarrAccess.default() so the
two fakes can be used together for integration-style tool tests.
"""

from dataclasses import dataclass, field

import numpy as np

from cell_explorer_agent.tools.strata_protocol import (
    AtomicStrataResult,
    CoarseStrataResult,
)


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
    atomic_table: AtomicStrataResult | None = None

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
                baseline = float(sub.min()) if sub.size > 0 else 0.0
                nnz[s_idx, g_idx] = int((sub > baseline).sum())

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

    @classmethod
    def with_atomic(
        cls,
        axes: list[str],
        stratum_values: dict[str, list[str]] | None = None,
        var_names: list[str] | None = None,
    ) -> "FakeStrataAccess":
        """Atomic-only fixture with the given axes.

        Stratum values per axis come from `stratum_values` when provided.
        Axes without explicit values get synthetic `<axis>_a` / `<axis>_b`
        labels — fine for routing tests that just assert `method ==
        "via_atomic_strata"` and don't query real-looking groups on that axis.

        For routing tests that query real group values (e.g. "T cell" vs
        "B cell" on the `cell_type` axis), pass them via stratum_values
        so the atomic table actually contains matching rows.
        """
        if not axes:
            raise ValueError("axes must be non-empty")
        n_genes = len(var_names) if var_names is not None else 50
        stratum_values = stratum_values or {}
        per_axis_values = [
            stratum_values.get(a, [f"{a}_a", f"{a}_b"]) for a in axes
        ]
        rows = _cartesian_product(per_axis_values)
        stratum_keys = np.array(rows, dtype=object)
        n_strata = stratum_keys.shape[0]
        n_cells = np.full(n_strata, 10, dtype=np.int32)
        sum_x = np.full((n_strata, n_genes), 1.0, dtype=np.float32)
        sum_xx = np.full((n_strata, n_genes), 1.0, dtype=np.float32)
        nnz = np.full((n_strata, n_genes), 5, dtype=np.int32)
        return cls(atomic_table=AtomicStrataResult(
            axes=list(axes),
            stratum_keys=stratum_keys,
            sum_x=sum_x,
            sum_xx=sum_xx,
            nnz=nnz,
            n_cells=n_cells,
        ))

    @classmethod
    def from_zarr_data_atomic(cls, fake_zarr, extra_axis: str = "synth") -> "FakeStrataAccess":
        """Build an atomic strata (cell_type × extra_axis) whose sums match
        fake_zarr.expression exactly, partitioned by a synthetic axis.

        Used by cross-path equivalence tests for the atomic fallback: the
        atomic table has MORE rows than a coarse on cell_type (because of
        the extra axis), so aggregation across the extra axis is exercised
        — and the aggregated result must match what X-scan produces.
        """
        col = fake_zarr.obs["cell_type"]
        cats = list(col.categories)
        codes = col.values
        n_obs = fake_zarr.n_obs
        # Synthetic axis: first half of cells = 'p', second half = 'q'.
        synth = np.array(["p"] * (n_obs // 2) + ["q"] * (n_obs - n_obs // 2))

        # Enumerate non-empty (cell_type, synth) buckets.
        buckets: list[tuple[str, str, np.ndarray]] = []
        for s_idx, cat_name in enumerate(cats):
            for synth_val in ("p", "q"):
                mask = (codes == s_idx) & (synth == synth_val)
                if mask.any():
                    buckets.append((cat_name, synth_val, mask))

        n_strata = len(buckets)
        n_genes = len(fake_zarr.var)
        sum_x = np.zeros((n_strata, n_genes), dtype="float64")
        sum_xx = np.zeros((n_strata, n_genes), dtype="float64")
        nnz = np.zeros((n_strata, n_genes), dtype="int32")
        n_cells = np.zeros(n_strata, dtype="int32")
        for s_idx, (_ct, _synth, mask) in enumerate(buckets):
            n_cells[s_idx] = int(mask.sum())
            for g_idx, gene in enumerate(fake_zarr.var):
                sub = fake_zarr.expression[gene][mask].astype("float64", copy=False)
                sum_x[s_idx, g_idx] = float(sub.sum())
                sum_xx[s_idx, g_idx] = float((sub * sub).sum())
                baseline = float(sub.min()) if sub.size > 0 else 0.0
                nnz[s_idx, g_idx] = int((sub > baseline).sum())

        stratum_keys = np.array(
            [[ct, synth_val] for ct, synth_val, _ in buckets], dtype=object,
        )
        return cls(atomic_table=AtomicStrataResult(
            axes=["cell_type", extra_axis],
            stratum_keys=stratum_keys,
            sum_x=sum_x,
            sum_xx=sum_xx,
            nnz=nnz,
            n_cells=n_cells,
        ))

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

    def has_atomic(self) -> bool:
        return self.atomic_table is not None

    def atomic_axes(self) -> list[str] | None:
        return list(self.atomic_table.axes) if self.atomic_table else None

    async def read_atomic(self) -> AtomicStrataResult:
        if self.atomic_table is None:
            raise ValueError("no atomic table")
        return self.atomic_table

    async def read_atomic_stratum_keys(self):
        if self.atomic_table is None:
            raise ValueError("no atomic table")
        return self.atomic_table.stratum_keys, self.atomic_table.n_cells

    async def read_atomic_rows(self, row_indices: list[int]) -> AtomicStrataResult:
        if self.atomic_table is None:
            raise ValueError("no atomic table")
        t = self.atomic_table
        if not row_indices:
            n_genes = t.sum_x.shape[1] if t.sum_x.ndim == 2 else 0
            return AtomicStrataResult(
                axes=list(t.axes),
                stratum_keys=t.stratum_keys[:0],
                sum_x=np.zeros((0, n_genes), dtype=t.sum_x.dtype),
                sum_xx=np.zeros((0, n_genes), dtype=t.sum_xx.dtype),
                nnz=np.zeros((0, n_genes), dtype=t.nnz.dtype),
                n_cells=t.n_cells[:0],
            )
        idx = np.asarray(row_indices, dtype=np.int64)
        return AtomicStrataResult(
            axes=list(t.axes),
            stratum_keys=t.stratum_keys[idx],
            sum_x=t.sum_x[idx],
            sum_xx=t.sum_xx[idx],
            nnz=t.nnz[idx],
            n_cells=t.n_cells[idx],
        )


def _cartesian_product(per_axis_values: list[list[str]]) -> list[list[str]]:
    """Return all combinations of values across the given axes."""
    rows: list[list[str]] = [[]]
    for values in per_axis_values:
        rows = [r + [v] for r in rows for v in values]
    return rows
