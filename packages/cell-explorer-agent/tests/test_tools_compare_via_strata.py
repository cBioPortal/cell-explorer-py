"""Unit tests for compare_groups_via_strata tool."""

import math

from cell_explorer_agent.tools.data.compare_via_strata import (
    compare_groups_via_strata_tool,
)
from tests.fakes.fake_strata import FakeStrataAccess


async def test_happy_path_top_genes_ranked_by_absolute_lfc(fake_zarr):
    """Top genes should be CD8A (T-cell marker) and MS4A1 (B-cell marker) —
    those are the two with biggest abs(LFC) in the FakeStrataAccess default."""
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="T cell", group_b="B cell", n=5)

    assert result["group_a"] == "T cell"
    assert result["group_b"] == "B cell"
    assert result["method"] == "log2_fold_change_via_coarse_strata"
    assert len(result["genes"]) == 5
    top_two = {g["symbol"] for g in result["genes"][:2]}
    # CD8A is gene 0 (high in T cell), MS4A1 is gene 2 (high in B cell)
    assert top_two == {"CD8A", "MS4A1"}


async def test_no_matching_coarse_returns_error(fake_zarr):
    """If no coarse table covers the requested obs_column, return an error
    instructing the LLM to fall back to compare_groups."""
    strata = FakeStrataAccess.default()  # only has 'cell_type' coarse
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="donor", group_a="d1", group_b="d2")

    assert "error" in result
    assert "no coarse strata table covers axes" in result["error"]
    assert "donor" in result["error"]
    assert "compare_groups" in result["error"]


async def test_unknown_group_a_returns_error(fake_zarr):
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="Nope", group_b="B cell")
    assert "error" in result
    assert "Nope" in result["error"]


async def test_unknown_group_b_returns_error(fake_zarr):
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="T cell", group_b="Nope")
    assert "error" in result
    assert "Nope" in result["error"]


async def test_zero_cells_returns_error(fake_zarr):
    """If a stratum has n_cells=0 (shouldn't happen in real data but defensive)."""
    strata = FakeStrataAccess.default()
    strata.coarse_tables["cell_type"].n_cells[1] = 0  # zero out B cell count
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="T cell", group_b="B cell")
    assert "error" in result
    assert "0 cells" in result["error"]


async def test_n_is_capped_at_hard_limit(fake_zarr):
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="T cell", group_b="B cell", n=999)
    # HARD_LIMIT = 50; fake has 50 genes so we get all 50
    assert len(result["genes"]) == 50


async def test_tool_metadata(fake_zarr):
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    assert tool.name == "compare_groups_via_strata"
    assert tool.kind == "data"
    # Args schema mirrors compare_groups exactly
    props = tool.args_schema["properties"]
    assert set(props.keys()) == {"obs_column", "group_a", "group_b", "n"}
    assert tool.args_schema["required"] == ["obs_column", "group_a", "group_b"]


async def test_output_genes_have_expected_keys(fake_zarr):
    strata = FakeStrataAccess.default()
    tool = compare_groups_via_strata_tool(fake_zarr, strata, limit_bytes=32_768)
    result = await tool.func(obs_column="cell_type", group_a="T cell", group_b="B cell", n=3)
    gene = result["genes"][0]
    assert set(gene.keys()) == {"symbol", "log2_fold_change", "mean_a", "mean_b"}
    # CD8A T-cell mean: 200/34 ≈ 5.88; B-cell mean: 20/33 ≈ 0.606
    if gene["symbol"] == "CD8A":
        assert math.isclose(gene["mean_a"], 200/34, rel_tol=1e-5)
        assert math.isclose(gene["mean_b"], 20/33, rel_tol=1e-5)
