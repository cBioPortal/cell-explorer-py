from cell_explorer_agent.tools.ui_action.embedding import set_embedding_tool
from cell_explorer_agent.tools.ui_action.color import (
    set_color_by_gene_tool,
    set_color_by_category_tool,
)
from cell_explorer_agent.tools.ui_action.filter import (
    filter_by_ids_tool,
    clear_filter_tool,
)
from cell_explorer_agent.tools.ui_action.summary import (
    add_summary_obs_column_tool,
    add_summary_gene_tool,
)
from cell_explorer_agent.tools.ui_action.viewport import set_viewport_tool
from cell_explorer_agent.tools.ui_action.summary_context import set_summary_context_tool
from cell_explorer_agent.tools.ui_action.gene_label_column import set_gene_label_column_tool


async def test_set_embedding_valid(fake_zarr):
    tool = set_embedding_tool(fake_zarr)
    r = await tool.func(embedding="X_umap")
    assert r == {"payload": {"embedding": "X_umap"}}
    assert tool.kind == "ui_action"


async def test_set_embedding_unknown(fake_zarr):
    tool = set_embedding_tool(fake_zarr)
    r = await tool.func(embedding="X_nothing")
    assert "error" in r


async def test_set_color_by_gene_valid(fake_zarr):
    tool = set_color_by_gene_tool(fake_zarr)
    r = await tool.func(gene="CD8A")
    assert r == {"payload": {"colorBy": "gene", "gene": "CD8A"}}


async def test_set_color_by_gene_unknown(fake_zarr):
    tool = set_color_by_gene_tool(fake_zarr)
    r = await tool.func(gene="FAKE")
    assert "error" in r


async def test_set_color_by_category_valid(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    r = await tool.func(category="cell_type")
    assert r == {"payload": {"colorBy": "category", "category": "cell_type"}}


async def test_set_color_by_category_unknown(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    r = await tool.func(category="nope")
    assert "error" in r


async def test_filter_by_ids_valid(fake_zarr):
    tool = filter_by_ids_tool(fake_zarr, filter_ids_max=100_000)
    r = await tool.func(obs_column="cell_type", ids=["cell0", "cell1"])
    assert r["payload"]["filter"]["obsColumn"] == "cell_type"
    assert r["payload"]["filter"]["ids"] == ["cell0", "cell1"]


async def test_filter_by_ids_too_many(fake_zarr):
    tool = filter_by_ids_tool(fake_zarr, filter_ids_max=10)
    r = await tool.func(obs_column="cell_type", ids=[f"c{i}" for i in range(100)])
    assert "error" in r
    assert "exceeds" in r["error"]


async def test_filter_by_ids_unknown_column(fake_zarr):
    tool = filter_by_ids_tool(fake_zarr, filter_ids_max=100)
    r = await tool.func(obs_column="nope", ids=["cell0"])
    assert "error" in r


async def test_clear_filter():
    tool = clear_filter_tool()
    r = await tool.func()
    assert r == {"payload": {"filter": {"obsColumn": "_none", "ids": []}}}


async def test_add_summary_obs_column_valid(fake_zarr):
    tool = add_summary_obs_column_tool(fake_zarr)
    r = await tool.func(obs_column="cell_type")
    assert r == {"payload": {"summaryObsColumns": ["cell_type"]}}


async def test_add_summary_obs_column_unknown(fake_zarr):
    tool = add_summary_obs_column_tool(fake_zarr)
    r = await tool.func(obs_column="nope")
    assert "error" in r


async def test_add_summary_gene_valid(fake_zarr):
    tool = add_summary_gene_tool(fake_zarr)
    r = await tool.func(gene="CD8A")
    assert r == {"payload": {"summaryGenes": ["CD8A"]}}


async def test_add_summary_gene_unknown(fake_zarr):
    tool = add_summary_gene_tool(fake_zarr)
    r = await tool.func(gene="NO_SUCH_GENE")
    assert "error" in r


async def test_set_viewport_valid():
    tool = set_viewport_tool()
    r = await tool.func(target_x=100.0, target_y=50.0, zoom=3.5)
    assert r == {"payload": {"viewport": {"target": [100.0, 50.0], "zoom": 3.5}}}
    assert tool.kind == "ui_action"


async def test_set_viewport_negative_zoom_ok():
    """Zoom can be negative (zoomed out beyond default fit-to-view)."""
    tool = set_viewport_tool()
    r = await tool.func(target_x=0.0, target_y=0.0, zoom=-1.0)
    assert r["payload"]["viewport"]["zoom"] == -1.0


async def test_set_summary_context_overall():
    tool = set_summary_context_tool()
    r = await tool.func(value="overall")
    assert r == {"payload": {"summaryContext": "overall"}}
    assert tool.kind == "ui_action"


async def test_set_summary_context_selections():
    tool = set_summary_context_tool()
    r = await tool.func(value="selections")
    assert r == {"payload": {"summaryContext": "selections"}}


async def test_set_summary_context_invalid_value():
    tool = set_summary_context_tool()
    r = await tool.func(value="other")
    assert "error" in r


async def test_set_gene_label_column_valid(fake_zarr):
    # FakeZarrAccess.default() includes 'gene_symbol' in var_columns_data
    tool = set_gene_label_column_tool(fake_zarr)
    r = await tool.func(column="gene_symbol")
    assert r == {"payload": {"geneLabelColumn": "gene_symbol"}}
    assert tool.kind == "ui_action"


async def test_set_gene_label_column_unknown(fake_zarr):
    tool = set_gene_label_column_tool(fake_zarr)
    r = await tool.func(column="not_a_column")
    assert "error" in r
