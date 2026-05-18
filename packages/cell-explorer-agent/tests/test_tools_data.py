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
    assert len(result["genes"]) == 5
    # CD8A should rank highly in T cells
    symbols = [g["symbol"] for g in result["genes"]]
    assert "CD8A" in symbols


from cell_explorer_agent.tools.data.compare import compare_groups_tool


async def test_compare_groups_returns_top_lfc(fake_zarr):
    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="B cell", n=5
    )
    assert len(result["genes"]) == 5
    symbols = [g["symbol"] for g in result["genes"]]
    assert "CD8A" in symbols  # T cell boost makes it a top marker
    # positive lfc means higher in group_a
    cd8a = next(g for g in result["genes"] if g["symbol"] == "CD8A")
    assert cd8a["log2_fold_change"] > 0


async def test_compare_groups_unknown_group(fake_zarr):
    tool = compare_groups_tool(fake_zarr, limit_bytes=32_768, concurrency=4)
    result = await tool.func(
        obs_column="cell_type", group_a="T cell", group_b="Nope", n=5
    )
    assert "error" in result


async def test_compare_groups_ranks_finite_above_non_finite_lfc():
    """Regression for #121: NaN and ±Inf LFC values must not displace finite
    top-magnitude entries in the ranking.

    Scaled expression (negative means) produces NaN LFC whenever
    (mean_a + PSEUDO) / (mean_b + PSEUDO) is negative. Means that exactly
    cancel the pseudocount produce ±Inf (log2(0)). Python's list.sort with
    NaN keys is undefined and was scattering NaN through the sorted list,
    burying the real top genes. The fix uses np.argsort with non-finite
    values mapped to a sort key of +inf so they always fall to the end.
    """
    import numpy as np

    from cell_explorer_agent.tools.zarr_protocol import ObsColumn
    from tests.fakes.fake_zarr import FakeZarrAccess

    n_obs = 40
    cats = ["T cell", "B cell"]
    codes = np.array([0] * 20 + [1] * 20, dtype=np.int8)
    obs_index = np.array([f"c{i}" for i in range(n_obs)])

    # 8 "NaN-producing" genes: opposite-sign means after pseudocount.
    expression: dict[str, np.ndarray] = {}
    nan_genes = [f"nan_gene_{i}" for i in range(8)]
    for g in nan_genes:
        arr = np.empty(n_obs, dtype="float32")
        arr[:20] = -0.5  # T cell
        arr[20:] = 0.5  # B cell
        expression[g] = arr

    # 2 "Inf-producing" genes: mean_a + PSEUDO == 0 exactly. We use float64
    # because float32(-1e-3) doesn't exactly round-trip to cancel PSEUDO=1e-3
    # in float64 arithmetic, so we'd get a tiny non-zero value instead of Inf.
    inf_genes = ["inf_gene_0", "inf_gene_1"]
    for g in inf_genes:
        expression[g] = np.concatenate([
            np.full(20, -1e-3, dtype="float64"),  # mean_a + PSEUDO == 0
            np.full(20, 0.5, dtype="float64"),
        ])

    # One "real top" gene with a clear large finite |LFC|
    expression["real_top"] = np.concatenate([
        np.full(20, -0.001, dtype="float32"),
        np.full(20, -0.142, dtype="float32"),
    ])

    # Three "background" genes with finite low |LFC|. They must rank above
    # any non-finite-LFC gene even though their |LFC| is small.
    background_genes = ["bg_0", "bg_1", "bg_2"]
    for i, g in enumerate(background_genes):
        expression[g] = np.concatenate([
            np.full(20, 0.10 + 0.01 * i, dtype="float32"),
            np.full(20, 0.12 + 0.01 * i, dtype="float32"),
        ])

    var = nan_genes + inf_genes + ["real_top"] + background_genes
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
    # 'real_top' has the largest finite |LFC| and must rank #1
    assert symbols[0] == "real_top", symbols
    # No non-finite-LFC gene should appear — there are 4 finite entries
    # available (real_top + 3 backgrounds), so n=4 must not include any
    # NaN- or Inf-LFC gene.
    assert not any(s.startswith("nan_gene_") for s in symbols), symbols
    assert not any(s.startswith("inf_gene_") for s in symbols), symbols
