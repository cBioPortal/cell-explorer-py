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
    def default(cls) -> "FakeStrataAccess":
        n_strata = 3
        n_genes = 50
        stratum_keys = np.array([["T cell"], ["B cell"], ["Monocyte"]])
        # cell_type cardinalities: arbitrary-but-fixed split summing to 100
        n_cells = np.array([34, 33, 33], dtype=np.int32)
        # Deterministic sums: each stratum gets distinct per-gene patterns
        sum_x = np.zeros((n_strata, n_genes), dtype=np.float32)
        # T-cell stratum: CD8A (gene 0) elevated so log-fold tests can rank it
        sum_x[0, 0] = 200.0   # T cell CD8A
        sum_x[1, 0] = 20.0    # B cell CD8A
        sum_x[2, 0] = 20.0    # Monocyte CD8A
        # MS4A1 (gene 2): elevated in B cells
        sum_x[0, 2] = 10.0
        sum_x[1, 2] = 150.0
        sum_x[2, 2] = 10.0
        # Background expression for the other genes
        for j in range(n_genes):
            if j in (0, 2):
                continue
            sum_x[:, j] = [50.0, 50.0, 50.0]
        sum_xx = sum_x * sum_x  # rough placeholder, not used by compare_groups path
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
