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
