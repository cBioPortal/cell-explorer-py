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
