from cell_explorer_agent.tools.data.schema import get_dataset_schema_tool


async def test_get_dataset_schema(fake_zarr):
    tool = get_dataset_schema_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func()

    assert result["n_obs"] == 100
    assert result["n_var"] == 50
    assert "cell_type" in [c["name"] for c in result["obs_columns"]]
    cell_type = next(c for c in result["obs_columns"] if c["name"] == "cell_type")
    assert cell_type["dtype"] == "categorical"
    assert cell_type["cardinality"] == 3
    assert "X_umap" in result["embeddings"]
    assert result["var_count"] == 50
    # var_columns lists names from the var dataframe (e.g. gene_symbol, feature_id)
    assert "var_columns" in result
    assert "feature_id" in result["var_columns"]
    assert "gene_symbol" in result["var_columns"]


async def test_get_dataset_schema_is_data_kind(fake_zarr):
    tool = get_dataset_schema_tool(fake_zarr, limit_bytes=32_768)
    assert tool.kind == "data"
    assert tool.name == "get_dataset_schema"


from cell_explorer_agent.tools.data.obs import describe_obs_column_tool


async def test_describe_categorical(fake_zarr):
    tool = describe_obs_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="cell_type")
    assert result["dtype"] == "categorical"
    assert result["total"] == 100
    assert len(result["top_categories"]) <= 50
    names = {c["value"] for c in result["top_categories"]}
    assert names == {"T cell", "B cell", "Monocyte"}


async def test_describe_numeric(fake_zarr):
    tool = describe_obs_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="n_counts")
    assert result["dtype"] == "numeric"
    assert set(result["stats"]) == {"min", "max", "mean", "median", "q1", "q3", "stddev"}
    assert result["stats"]["min"] < result["stats"]["max"]


async def test_describe_unknown_column_returns_error(fake_zarr):
    tool = describe_obs_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="nonexistent")
    assert "error" in result
    assert "nonexistent" in result["error"]


from cell_explorer_agent.tools.data.obs import cluster_stats_tool


async def test_cluster_stats(fake_zarr):
    tool = cluster_stats_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type")
    assert result["total"] == 100
    assert sum(g["count"] for g in result["groups"]) == 100
    names = {g["value"] for g in result["groups"]}
    assert names == {"T cell", "B cell", "Monocyte"}


async def test_cluster_stats_rejects_numeric(fake_zarr):
    tool = cluster_stats_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(obs_column="n_counts")
    assert "error" in result


from cell_explorer_agent.tools.data.genes import (
    search_genes_tool,
    gene_expression_summary_tool,
    top_expressed_genes_tool,
)


async def test_search_genes_substring(fake_zarr):
    tool = search_genes_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(query="CD", limit=20)
    symbols = {r["symbol"] for r in result["matches"]}
    assert "CD8A" in symbols
    assert "CD4" in symbols


async def test_search_genes_limit_capped(fake_zarr):
    tool = search_genes_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(query="gene", limit=1000)
    assert len(result["matches"]) <= 50  # hard cap
    assert result["capped"] is True


async def test_gene_expression_summary_overall(fake_zarr):
    tool = gene_expression_summary_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(gene="CD8A")
    assert result["gene"] == "CD8A"
    assert "overall" in result
    assert 0.0 <= result["overall"]["fraction_expressing"] <= 1.0
    assert result["overall"]["mean"] > 0


async def test_gene_expression_summary_grouped(fake_zarr):
    tool = gene_expression_summary_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(gene="CD8A", group_by="cell_type")
    groups = {g["value"]: g for g in result["per_group"]}
    # T cell got a +5 boost in the fixture
    assert groups["T cell"]["mean"] > groups["B cell"]["mean"]


async def test_gene_expression_summary_unknown_gene(fake_zarr):
    tool = gene_expression_summary_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(gene="NOT_A_GENE")
    assert "error" in result


async def test_top_expressed_genes(fake_zarr):
    tool = top_expressed_genes_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=5)
    assert result["method"] == "via_xscan"
    assert result["obs_column"] == "cell_type"
    assert result["group_value"] == "T cell"
    assert result["n_cells"] > 0
    assert len(result["genes"]) == 5
    # CD8A should rank highly in T cells
    symbols = [g["symbol"] for g in result["genes"]]
    assert "CD8A" in symbols
    # New output fields are present per gene
    for g in result["genes"]:
        assert set(g) == {"symbol", "mean", "fraction_expressing"}
        assert 0.0 <= g["fraction_expressing"] <= 1.0


async def test_top_expressed_genes_uses_strata_when_available(fake_zarr):
    """When a coarse table covers obs_column, the strata fast path is used."""
    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.default(var_names=fake_zarr.var)
    tool = top_expressed_genes_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert "error" not in result, result
    assert result["method"] == "via_coarse_strata"
    assert len(result["genes"]) == 3


async def test_top_expressed_genes_falls_back_to_xscan_when_no_strata(fake_zarr):
    tool = top_expressed_genes_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert result["method"] == "via_xscan"


async def test_top_expressed_genes_uses_atomic_when_no_coarse_matches(fake_zarr):
    """No coarse but atomic covers obs_column -> via_atomic_strata."""
    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.with_atomic(
        axes=["cell_type", "synth"],
        stratum_values={
            "cell_type": ["T cell", "B cell", "Monocyte"],
            "synth": ["p", "q"],
        },
        var_names=fake_zarr.var,
    )
    tool = top_expressed_genes_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert "error" not in result, result
    assert result["method"] == "via_atomic_strata"


async def test_top_expressed_genes_strata_and_xscan_produce_identical_means(fake_zarr):
    """Cross-path equivalence: coarse strata path must produce per-gene means
    byte-identical to the X-scan path on the same data.
    """
    import numpy as np

    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.from_zarr_data(fake_zarr)

    tool_with = top_expressed_genes_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    tool_without = top_expressed_genes_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    args = dict(obs_column="cell_type", group_value="T cell", n=10)

    result_strata = await tool_with.func(**args)
    result_xscan = await tool_without.func(**args)

    assert result_strata["method"] == "via_coarse_strata"
    assert result_xscan["method"] == "via_xscan"
    assert result_strata["n_cells"] == result_xscan["n_cells"]

    strata_syms = [g["symbol"] for g in result_strata["genes"]]
    xscan_syms = [g["symbol"] for g in result_xscan["genes"]]
    assert strata_syms == xscan_syms

    for gs, gx in zip(result_strata["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(gs["mean"], gx["mean"], atol=1e-9)
        np.testing.assert_allclose(
            gs["fraction_expressing"], gx["fraction_expressing"], atol=1e-9,
        )


from cell_explorer_agent.tools.data.compare import compare_groups_tool


async def test_compare_groups_returns_top_by_cohens_d(fake_zarr):
    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=5
    )
    assert result["ranked_by"] == "cohens_d"
    assert result["method"] == "via_xscan"
    assert len(result["genes"]) == 5
    symbols = [g["symbol"] for g in result["genes"]]
    assert "CD8A" in symbols  # T cell boost makes it a top marker
    # positive cohens_d means higher in group_a
    cd8a = next(g for g in result["genes"] if g["symbol"] == "CD8A")
    assert cd8a["cohens_d"] > 0
    assert cd8a["t_statistic"] > 0
    # sufficient stats are exposed for downstream consumers
    for key in ("mean_a", "mean_b", "var_a", "var_b", "n_a", "n_b"):
        assert key in cd8a, key


async def test_compare_groups_unknown_group(fake_zarr):
    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="Nope", n=5
    )
    assert "error" in result


async def test_compare_groups_ranks_finite_above_non_finite_d():
    """Cohen's d is non-finite for zero-variance genes (NaN if both means
    equal, ±Inf otherwise). They must not displace finite top-magnitude
    entries in the ranking.
    """
    import numpy as np

    from cell_explorer_agent.tools.zarr_protocol import ObsColumn
    from tests.fakes.fake_zarr import FakeZarrAccess

    n_obs = 40
    cats = ["T cell", "B cell"]
    codes = np.array([0] * 20 + [1] * 20, dtype=np.int8)
    obs_index = np.array([f"c{i}" for i in range(n_obs)])

    # 8 zero-variance genes (constant within each group), unequal means
    # -> Inf cohens_d.
    expression: dict[str, np.ndarray] = {}
    inf_genes = [f"inf_gene_{i}" for i in range(8)]
    for g in inf_genes:
        arr = np.empty(n_obs, dtype="float64")
        arr[:20] = -0.5  # T cell constant
        arr[20:] = 0.5   # B cell constant
        expression[g] = arr

    # 2 zero-variance genes with equal means -> NaN cohens_d.
    nan_genes = ["nan_gene_0", "nan_gene_1"]
    for g in nan_genes:
        expression[g] = np.full(n_obs, 0.3, dtype="float64")

    # One "real top" gene with finite large |d| (different means + small variance).
    rng = np.random.default_rng(0)
    real_top = np.concatenate([
        rng.normal(loc=2.0, scale=0.05, size=20).astype("float32"),
        rng.normal(loc=-2.0, scale=0.05, size=20).astype("float32"),
    ])
    expression["real_top"] = real_top

    # Three "background" genes with finite small |d| (similar means + similar
    # variance). Must rank above any non-finite entry even though |d| is small.
    background_genes = ["bg_0", "bg_1", "bg_2"]
    for i, g in enumerate(background_genes):
        expression[g] = np.concatenate([
            rng.normal(loc=0.10 + 0.01 * i, scale=0.5, size=20).astype("float32"),
            rng.normal(loc=0.12 + 0.01 * i, scale=0.5, size=20).astype("float32"),
        ])

    var = inf_genes + nan_genes + ["real_top"] + background_genes
    fake = FakeZarrAccess(
        n_obs=n_obs,
        n_var=len(var),
        obs={
            "cell_type": ObsColumn(
                name="cell_type",
                dtype="categorical",
                values=codes,
                categories=cats,
                index=obs_index,
            ),
        },
        var=var,
        expression=expression,
    )

    tool = compare_groups_tool(fake, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=4
    )

    symbols = [g["symbol"] for g in result["genes"]]
    assert symbols[0] == "real_top", symbols
    assert not any(s.startswith("inf_gene_") for s in symbols), symbols
    assert not any(s.startswith("nan_gene_") for s in symbols), symbols


async def test_compare_groups_n_lt_2_returns_error():
    """Variance is undefined when a group has fewer than 2 cells."""
    import numpy as np

    from cell_explorer_agent.tools.zarr_protocol import ObsColumn
    from tests.fakes.fake_zarr import FakeZarrAccess

    n_obs = 10
    cats = ["A", "B"]
    # 9 cells in A, 1 cell in B
    codes = np.array([0] * 9 + [1] * 1, dtype=np.int8)
    obs_index = np.array([f"c{i}" for i in range(n_obs)])
    fake = FakeZarrAccess(
        n_obs=n_obs,
        n_var=1,
        obs={
            "cell_type": ObsColumn(
                name="cell_type",
                dtype="categorical",
                values=codes,
                categories=cats,
                index=obs_index,
            ),
        },
        var=["g0"],
        expression={"g0": np.zeros(n_obs, dtype="float32")},
    )

    tool = compare_groups_tool(fake, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", group_a="A", group_b="B")
    assert "error" in result
    assert "< 2 cells" in result["error"]


async def test_compare_groups_output_schema(fake_zarr):
    """Top-level + per-gene shapes must match the v1 spec exactly."""
    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert set(result) == {
        "obs_column", "group_a", "group_b", "ranked_by", "method", "genes",
    }
    assert result["ranked_by"] == "cohens_d"
    assert result["method"] == "via_xscan"
    for g in result["genes"]:
        assert set(g) == {
            "symbol", "cohens_d", "t_statistic",
            "mean_a", "mean_b", "var_a", "var_b", "n_a", "n_b",
        }, g


async def test_compare_groups_handles_zero_variance_genes(fake_zarr):
    """A gene that's constant within each group (var=0) must not crash and
    must not rank above the boosted-CD8A finite-stat gene.
    """
    import numpy as np

    # fake_zarr is function-scoped so this mutation doesn't leak between tests.
    # Replace MS4A1's expression with a constant-per-group array so it
    # produces non-finite Cohen's d (var = 0 within each group).
    arr = np.empty(fake_zarr.n_obs, dtype="float32")
    codes = fake_zarr.obs["cell_type"].values
    arr[codes == 0] = 1.0  # T cell
    arr[codes == 1] = 5.0  # B cell
    arr[codes == 2] = 3.0  # Monocyte
    fake_zarr.expression["MS4A1"] = arr

    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=5
    )
    # CD8A should still be the top marker (T cell boost is finite + large d)
    symbols = [g["symbol"] for g in result["genes"]]
    assert symbols[0] == "CD8A", symbols


async def test_compare_groups_uses_strata_when_available(fake_zarr):
    """When a coarse table covers obs_column, the strata fast path is used."""
    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.default(var_names=fake_zarr.var)
    tool = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert "error" not in result, result
    assert result["method"] == "via_coarse_strata"
    assert len(result["genes"]) == 3


async def test_compare_groups_falls_back_to_xscan_when_no_matching_coarse(fake_zarr):
    """Strata exists but covers a different axis -> X-scan fallback."""
    from tests.fakes.fake_strata import FakeStrataAccess

    # Build a strata where the only coarse axis is 'donor_id', not 'cell_type'.
    strata = FakeStrataAccess.with_axis(axis="donor_id", var_names=fake_zarr.var)
    tool = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert "error" not in result, result
    assert result["method"] == "via_xscan"


async def test_compare_groups_falls_back_to_xscan_when_no_strata(fake_zarr):
    """No strata wired in at all -> X-scan."""
    tool = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert result["method"] == "via_xscan"


async def test_compare_groups_strata_and_xscan_produce_identical_stats(fake_zarr):
    """Same query through both paths must produce the same per-gene stats.

    Forces the strata table to have sums computed from the SAME data the
    X-scan sees. Without this, drift between paths is invisible until a
    real-world A/B eval catches it (see #121's eval).
    """
    import numpy as np

    from tests.fakes.fake_strata import FakeStrataAccess

    # Build a strata where sum_x / sum_xx for each stratum row are derived
    # directly from fake_zarr.expression, so by construction the two paths
    # are computing the same thing.
    strata = FakeStrataAccess.from_zarr_data(fake_zarr)

    tool_with = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    tool_without = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    args = dict(obs_column="cell_type", group_a="T cell", group_b="B cell", n=10)

    result_strata = await tool_with.func(**args)
    result_xscan = await tool_without.func(**args)

    assert result_strata["method"] == "via_coarse_strata"
    assert result_xscan["method"] == "via_xscan"

    # Same top-N genes in the same order.
    strata_syms = [g["symbol"] for g in result_strata["genes"]]
    xscan_syms = [g["symbol"] for g in result_xscan["genes"]]
    assert strata_syms == xscan_syms

    # Per-gene stats match to within float tolerance.
    for gs, gx in zip(result_strata["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(gs["cohens_d"], gx["cohens_d"], atol=1e-9)
        np.testing.assert_allclose(gs["t_statistic"], gx["t_statistic"], atol=1e-9)
        np.testing.assert_allclose(gs["mean_a"], gx["mean_a"], atol=1e-9)
        np.testing.assert_allclose(gs["mean_b"], gx["mean_b"], atol=1e-9)
        np.testing.assert_allclose(gs["var_a"], gx["var_a"], atol=1e-9)
        np.testing.assert_allclose(gs["var_b"], gx["var_b"], atol=1e-9)


async def test_compare_groups_uses_atomic_when_no_coarse_matches(fake_zarr):
    """When no coarse covers obs_column but atomic does, the atomic
    fast path engages (method='via_atomic_strata') and aggregation across
    the atomic table's other axes produces a result.
    """
    from tests.fakes.fake_strata import FakeStrataAccess

    # Atomic covers (cell_type, synth) but there is NO coarse for cell_type.
    strata = FakeStrataAccess.with_atomic(
        axes=["cell_type", "synth"],
        stratum_values={
            "cell_type": ["T cell", "B cell", "Monocyte"],
            "synth": ["p", "q"],
        },
        var_names=fake_zarr.var,
    )
    tool = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert "error" not in result, result
    assert result["method"] == "via_atomic_strata"
    assert len(result["genes"]) == 3


async def test_compare_groups_falls_back_to_xscan_when_atomic_doesnt_cover_obs_column(fake_zarr):
    """Atomic exists but doesn't include obs_column -> X-scan fallback."""
    from tests.fakes.fake_strata import FakeStrataAccess

    # Atomic covers ('donor', 'phase') only; the query is on 'cell_type'.
    strata = FakeStrataAccess.with_atomic(
        axes=["donor", "phase"],
        var_names=fake_zarr.var,
    )
    tool = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=3
    )
    assert "error" not in result, result
    assert result["method"] == "via_xscan"


async def test_compare_groups_atomic_and_xscan_produce_identical_stats(fake_zarr):
    """The atomic-aggregation path must produce stats identical to X-scan
    on the same data. Catches drift between atomic-aggregation and the
    direct expression scan.
    """
    import numpy as np

    from tests.fakes.fake_strata import FakeStrataAccess

    # Build an atomic strata whose sums come directly from fake_zarr.expression,
    # partitioned by (cell_type, synth) — so aggregation across the synth
    # axis is exercised, not skipped.
    strata = FakeStrataAccess.from_zarr_data_atomic(fake_zarr)

    tool_with = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    tool_without = compare_groups_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    args = dict(obs_column="cell_type", group_a="T cell", group_b="B cell", n=10)

    result_atomic = await tool_with.func(**args)
    result_xscan = await tool_without.func(**args)

    assert result_atomic["method"] == "via_atomic_strata"
    assert result_xscan["method"] == "via_xscan"

    atomic_syms = [g["symbol"] for g in result_atomic["genes"]]
    xscan_syms = [g["symbol"] for g in result_xscan["genes"]]
    assert atomic_syms == xscan_syms

    for ga, gx in zip(result_atomic["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(ga["cohens_d"], gx["cohens_d"], atol=1e-9)
        np.testing.assert_allclose(ga["t_statistic"], gx["t_statistic"], atol=1e-9)
        np.testing.assert_allclose(ga["mean_a"], gx["mean_a"], atol=1e-9)
        np.testing.assert_allclose(ga["mean_b"], gx["mean_b"], atol=1e-9)
        np.testing.assert_allclose(ga["var_a"], gx["var_a"], atol=1e-9)
        np.testing.assert_allclose(ga["var_b"], gx["var_b"], atol=1e-9)


from cell_explorer_agent.tools.data.markers import find_markers_tool


async def test_find_markers_returns_top_by_cohens_d(fake_zarr):
    tool = find_markers_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=5)
    assert result["method"] == "via_xscan"
    assert result["ranked_by"] == "cohens_d"
    assert result["obs_column"] == "cell_type"
    assert result["group_value"] == "T cell"
    assert result["n_cells_group"] > 0
    assert result["n_cells_rest"] > 0
    assert len(result["genes"]) == 5
    # CD8A is the T-cell-boosted marker in the fixture, should rank top
    symbols = [g["symbol"] for g in result["genes"]]
    assert "CD8A" in symbols
    cd8a = next(g for g in result["genes"] if g["symbol"] == "CD8A")
    # T cells are 'group', everything else is 'rest'; CD8A should be higher in group
    assert cd8a["cohens_d"] > 0


async def test_find_markers_output_schema(fake_zarr):
    """Top-level + per-gene shapes."""
    tool = find_markers_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert set(result) == {
        "obs_column", "group_value", "ranked_by", "method",
        "n_cells_group", "n_cells_rest", "genes",
    }
    for g in result["genes"]:
        assert set(g) == {
            "symbol", "cohens_d", "t_statistic",
            "mean_group", "mean_rest", "var_group", "var_rest",
            "n_group", "n_rest",
        }


async def test_find_markers_uses_coarse_strata_when_available(fake_zarr):
    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.default(var_names=fake_zarr.var)
    tool = find_markers_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert "error" not in result, result
    assert result["method"] == "via_coarse_strata"
    assert len(result["genes"]) == 3


async def test_find_markers_uses_atomic_when_no_coarse(fake_zarr):
    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.with_atomic(
        axes=["cell_type", "synth"],
        stratum_values={
            "cell_type": ["T cell", "B cell", "Monocyte"],
            "synth": ["p", "q"],
        },
        var_names=fake_zarr.var,
    )
    tool = find_markers_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert "error" not in result, result
    assert result["method"] == "via_atomic_strata"


async def test_find_markers_falls_back_to_xscan(fake_zarr):
    tool = find_markers_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert result["method"] == "via_xscan"


async def test_find_markers_strata_and_xscan_produce_identical_stats(fake_zarr):
    """Cross-path equivalence regression guard."""
    import numpy as np

    from tests.fakes.fake_strata import FakeStrataAccess

    strata = FakeStrataAccess.from_zarr_data(fake_zarr)

    tool_with = find_markers_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=strata,
    )
    tool_without = find_markers_tool(
        fake_zarr, limit_bytes=32_768, concurrency=4, strata=None,
    )
    args = dict(obs_column="cell_type", group_value="T cell", n=10)

    result_strata = await tool_with.func(**args)
    result_xscan = await tool_without.func(**args)

    assert result_strata["method"] == "via_coarse_strata"
    assert result_xscan["method"] == "via_xscan"
    assert result_strata["n_cells_group"] == result_xscan["n_cells_group"]
    assert result_strata["n_cells_rest"] == result_xscan["n_cells_rest"]

    strata_syms = [g["symbol"] for g in result_strata["genes"]]
    xscan_syms = [g["symbol"] for g in result_xscan["genes"]]
    assert strata_syms == xscan_syms

    for gs, gx in zip(result_strata["genes"], result_xscan["genes"]):
        np.testing.assert_allclose(gs["cohens_d"], gx["cohens_d"], atol=1e-9)
        np.testing.assert_allclose(gs["t_statistic"], gx["t_statistic"], atol=1e-9)
        np.testing.assert_allclose(gs["mean_group"], gx["mean_group"], atol=1e-9)
        np.testing.assert_allclose(gs["mean_rest"], gx["mean_rest"], atol=1e-9)
        np.testing.assert_allclose(gs["var_group"], gx["var_group"], atol=1e-9)
        np.testing.assert_allclose(gs["var_rest"], gx["var_rest"], atol=1e-9)


async def test_find_markers_group_not_found_returns_error(fake_zarr):
    tool = find_markers_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", group_value="Not_A_Group")
    assert "error" in result
    assert "Not_A_Group" in result["error"]


async def test_top_expressed_genes_includes_chart_hint(fake_zarr):
    """top_expressed_genes returns a chart hint of type top_genes_bar."""
    from cell_explorer_agent.config import AgentConfig

    cfg = AgentConfig()
    tool = top_expressed_genes_tool(
        fake_zarr,
        limit_bytes=cfg.tool_result_max_bytes,
        concurrency=cfg.gene_scan_concurrency,
    )
    result = await tool.func(obs_column="cell_type", group_value="T cell", n=3)
    assert "chart" in result
    assert result["chart"]["type"] == "top_genes_bar"
    assert "data" in result["chart"]
    # chart.data echoes the top-genes list — same shape as result["genes"]
    # so the frontend renderer can consume it directly.
    assert result["chart"]["data"]["genes"] == result["genes"]


async def test_xscan_panel_sums_shape(fake_zarr):
    """_xscan_panel_sums returns (n_genes, n_categories) matrices for sum_x and nnz, plus n_cells per category."""
    import numpy as np
    from cell_explorer_agent.tools.data.genes import _xscan_panel_sums

    col = await fake_zarr.obs_column("cell_type")
    # Use whatever categories the fixture has — first 2 of them.
    cat_codes = list(range(min(2, len(col.categories))))
    masks = [col.values == c for c in cat_codes]
    # Use the first 3 gene names from var.
    var_names = await fake_zarr.var_names()
    gene_names = list(var_names[:3])

    sum_x, nnz, n_cells = await _xscan_panel_sums(
        fake_zarr, gene_names, masks, concurrency=4,
    )
    assert sum_x.shape == (3, 2)
    assert nnz.shape == (3, 2)
    assert n_cells.shape == (2,)
    assert n_cells[0] == int(masks[0].sum())
    assert n_cells[1] == int(masks[1].sum())
    # Sanity: nnz <= n_cells for each (gene, category)
    for j in range(2):
        for i in range(3):
            assert nnz[i, j] <= n_cells[j]


async def test_gene_panel_by_obs_returns_matrix(fake_zarr):
    """gene_panel_by_obs returns matrix-format data + gene_panel_dotplot chart hint."""
    from cell_explorer_agent.config import AgentConfig
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool

    cfg = AgentConfig()
    tool = gene_panel_by_obs_tool(
        fake_zarr,
        limit_bytes=cfg.tool_result_max_bytes,
        concurrency=cfg.gene_scan_concurrency,
    )
    result = await tool.func(obs_column="cell_type", genes=["CD8A", "CD4"])

    assert "data" in result
    data = result["data"]
    assert data["genes"] == ["CD8A", "CD4"]
    # All fixture categories preserved in dataset order.
    assert "T cell" in data["categories"]
    assert "B cell" in data["categories"]
    # Matrices: rows = genes, cols = categories.
    assert len(data["mean"]) == 2  # 2 genes
    assert len(data["mean"][0]) == len(data["categories"])
    assert len(data["frac_expressing"]) == 2
    assert len(data["n_cells_per_category"]) == len(data["categories"])
    # Chart hint with the same data echoed for the renderer.
    assert result["chart"]["type"] == "gene_panel_dotplot"
    assert result["chart"]["data"]["genes"] == ["CD8A", "CD4"]


async def test_gene_panel_by_obs_validates_obs_column(fake_zarr):
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool
    tool = gene_panel_by_obs_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="nonexistent_column", genes=["CD8A"])
    assert "error" in result
    assert "nonexistent_column" in result["error"]


async def test_gene_panel_by_obs_rejects_non_categorical(fake_zarr):
    """A continuous obs column can't be used as a category axis."""
    import pytest
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool
    tool = gene_panel_by_obs_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    cols = await fake_zarr.obs_columns()
    continuous = next((c for c in cols if c.dtype != "categorical"), None)
    if continuous is None:
        pytest.skip("fake_zarr fixture has no continuous obs column")
    result = await tool.func(obs_column=continuous.name, genes=["CD8A"])
    assert "error" in result


async def test_gene_panel_by_obs_unknown_gene_error(fake_zarr):
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool
    tool = gene_panel_by_obs_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", genes=["NOT_A_REAL_GENE"])
    assert "error" in result


async def test_gene_panel_by_obs_caps_gene_count(fake_zarr):
    """Tool rejects >50 genes."""
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool
    tool = gene_panel_by_obs_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    too_many = [f"g{i}" for i in range(60)]
    result = await tool.func(obs_column="cell_type", genes=too_many)
    assert "error" in result


async def test_gene_panel_by_obs_empty_gene_list_errors(fake_zarr):
    from cell_explorer_agent.tools.data.genes import gene_panel_by_obs_tool
    tool = gene_panel_by_obs_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(obs_column="cell_type", genes=[])
    assert "error" in result


# ---------------------------------------------------------------------------
# Baseline-aware fraction_expressing tests (fix for floored/non-zero-minimum data)
# ---------------------------------------------------------------------------

async def test_xscan_panel_sums_floored_data_baseline():
    """_xscan_panel_sums uses min-as-baseline, not np.count_nonzero.

    expr = [0.5, 0.5, 2.0, 0.5, 3.5]
    masks: cat0 = first 2 cells, cat1 = last 3 cells
    union baseline = min(expr[union]) = 0.5
    expressing = val > 0.5
      cat0: [0.5, 0.5] -> 0 cells expressing
      cat1: [2.0, 0.5, 3.5] -> 2 cells expressing

    Old np.count_nonzero would give [2, 3] (all values nonzero).
    """
    import numpy as np

    from cell_explorer_agent.tools.data.genes import _xscan_panel_sums
    from cell_explorer_agent.tools.zarr_protocol import ObsColumn
    from tests.fakes.fake_zarr import FakeZarrAccess

    expr_arr = np.array([0.5, 0.5, 2.0, 0.5, 3.5], dtype="float32")
    fake = FakeZarrAccess(
        n_obs=5,
        n_var=1,
        obs={},
        var=["g0"],
        expression={"g0": expr_arr},
    )

    masks = [
        np.array([True, True, False, False, False]),
        np.array([False, False, True, True, True]),
    ]

    sum_x, nnz, n_cells = await _xscan_panel_sums(fake, ["g0"], masks, concurrency=1)

    assert sum_x.shape == (1, 2)
    assert nnz.shape == (1, 2)
    # sum_x unaffected by baseline change
    np.testing.assert_allclose(sum_x[0, 0], 1.0, atol=1e-6)   # 0.5+0.5
    np.testing.assert_allclose(sum_x[0, 1], 6.0, atol=1e-6)   # 2.0+0.5+3.5
    # baseline-aware counts
    assert nnz[0, 0] == 0, f"expected 0, got {nnz[0,0]}"  # no val > 0.5 in cat0
    assert nnz[0, 1] == 2, f"expected 2, got {nnz[0,1]}"  # 2.0 and 3.5 > 0.5


async def test_xscan_group_sums_floored_data_baseline():
    """_xscan_group_sums uses min-as-baseline over the masked cells.

    expr = [0.1, 0.1, 1.0, 0.1, 2.5], mask = all True
    baseline = 0.1; expressing = val > 0.1 -> 2 cells (1.0 and 2.5)
    Old np.count_nonzero would give 5 (all nonzero).
    """
    import numpy as np

    from cell_explorer_agent.tools.data.genes import _xscan_group_sums
    from tests.fakes.fake_zarr import FakeZarrAccess

    expr_arr = np.array([0.1, 0.1, 1.0, 0.1, 2.5], dtype="float32")
    fake = FakeZarrAccess(
        n_obs=5,
        n_var=1,
        obs={},
        var=["g0"],
        expression={"g0": expr_arr},
    )

    mask = np.ones(5, dtype=bool)
    sum_x, nnz = await _xscan_group_sums(fake, ["g0"], mask, concurrency=1)

    np.testing.assert_allclose(sum_x[0], 3.8, atol=1e-4)  # 0.1+0.1+1.0+0.1+2.5 (float32 precision)
    assert nnz[0] == 2, f"expected 2, got {nnz[0]}"  # only 1.0 and 2.5 > baseline 0.1


async def test_xscan_group_sums_zero_floored_backward_compat():
    """When true zeros exist, baseline=0 and nnz matches np.count_nonzero behavior.

    expr = [0, 0, 1, 2, 0], mask = all True
    baseline = 0; expressing = val > 0 -> 2 cells
    """
    import numpy as np

    from cell_explorer_agent.tools.data.genes import _xscan_group_sums
    from tests.fakes.fake_zarr import FakeZarrAccess

    expr_arr = np.array([0, 0, 1, 2, 0], dtype="float32")
    fake = FakeZarrAccess(
        n_obs=5,
        n_var=1,
        obs={},
        var=["g0"],
        expression={"g0": expr_arr},
    )

    mask = np.ones(5, dtype=bool)
    sum_x, nnz = await _xscan_group_sums(fake, ["g0"], mask, concurrency=1)

    np.testing.assert_allclose(sum_x[0], 3.0, atol=1e-6)
    assert nnz[0] == 2, f"expected 2, got {nnz[0]}"  # matches count_nonzero for zero-floored data


# ---------------------------------------------------------------------------
# describe_var_column tests — mirror the obs side
# ---------------------------------------------------------------------------

from cell_explorer_agent.tools.data.var import describe_var_column_tool
from cell_explorer_agent.tools.zarr_protocol import ObsColumn


async def test_describe_var_column_categorical(fake_zarr):
    import numpy as np

    # Seed the fake with a categorical var column (e.g., feature_type with two values)
    n_var = 50
    cats = ["protein_coding", "lncRNA"]
    codes = np.array([0] * 40 + [1] * 10, dtype=np.int32)
    fake_zarr.var_data["feature_type"] = ObsColumn(
        name="feature_type",
        dtype="categorical",
        values=codes,
        categories=list(cats),
    )
    fake_zarr.var_columns_data = ["feature_id", "gene_symbol", "feature_type"]

    tool = describe_var_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="feature_type")
    assert result["dtype"] == "categorical"
    assert result["total"] == n_var
    names = {c["value"] for c in result["top_categories"]}
    assert names == {"protein_coding", "lncRNA"}


async def test_describe_var_column_numeric(fake_zarr):
    import numpy as np

    n_var = 50
    fake_zarr.var_data["mean_counts"] = ObsColumn(
        name="mean_counts",
        dtype="numeric",
        values=np.linspace(0.1, 10.0, n_var, dtype=np.float32),
    )

    tool = describe_var_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="mean_counts")
    assert result["dtype"] == "numeric"
    assert result["total"] == n_var
    assert set(result["stats"]) == {"min", "max", "mean", "median", "q1", "q3", "stddev"}
    assert result["stats"]["min"] < result["stats"]["max"]


async def test_describe_var_column_unknown_returns_error(fake_zarr):
    tool = describe_var_column_tool(fake_zarr, limit_bytes=32_768)
    result = await tool.func(name="nonexistent_column")
    assert "error" in result
    assert "nonexistent_column" in result["error"]
