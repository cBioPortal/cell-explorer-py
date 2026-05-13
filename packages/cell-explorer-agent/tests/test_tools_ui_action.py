from cell_explorer_agent.tools.ui_action.embedding import set_embedding_tool
from cell_explorer_agent.tools.ui_action.color import (
    clear_color_by_tool,
    set_color_by_gene_tool,
    set_color_by_category_tool,
    set_color_scale_tool,
)
from cell_explorer_agent.tools.ui_action.filter import (
    filter_by_ids_tool,
    clear_filter_tool,
)
from cell_explorer_agent.tools.ui_action.filter_by_expression import (
    filter_by_gene_expression_tool,
)
from cell_explorer_agent.tools.ui_action.summary import (
    add_summary_obs_column_tool,
    add_summary_gene_tool,
    clear_summary_tool,
    remove_summary_gene_tool,
    remove_summary_obs_column_tool,
)
from cell_explorer_agent.tools.ui_action.viewport import (
    clear_viewport_tool,
    fit_viewport_to_selection_tool,
    set_viewport_tool,
)
from cell_explorer_agent.tools.ui_action.summary_context import set_summary_context_tool
from cell_explorer_agent.tools.ui_action.gene_label_column import set_gene_label_column_tool
from cell_explorer_agent.tools.ui_action.render import (
    clear_render_controls_tool,
    set_render_controls_tool,
)
from cell_explorer_agent.tools.ui_action.category_labels import (
    set_category_labels_tool,
)
from cell_explorer_agent.tools.ui_action.selection_display_mode import (
    set_selection_display_mode_tool,
)


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


async def test_set_color_by_category_with_highlight_valid(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    r = await tool.func(category="cell_type", highlight=["T cell"])
    assert r == {
        "payload": {
            "colorBy": "category",
            "category": "cell_type",
            "highlightedCategories": ["T cell"],
        }
    }


async def test_set_color_by_category_with_multiple_highlights(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    r = await tool.func(category="cell_type", highlight=["T cell", "Monocyte"])
    assert r["payload"]["highlightedCategories"] == ["T cell", "Monocyte"]


async def test_set_color_by_category_unknown_highlight(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    r = await tool.func(category="cell_type", highlight=["NotARealCategory"])
    assert "error" in r
    assert "NotARealCategory" in r["error"]


async def test_set_color_by_category_highlight_on_non_categorical(fake_zarr):
    tool = set_color_by_category_tool(fake_zarr)
    # n_counts is numeric in the fake fixture; can't highlight values on it
    r = await tool.func(category="n_counts", highlight=["100"])
    assert "error" in r
    assert "categorical" in r["error"]


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


async def test_filter_by_gene_expression_min_only(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    r = await tool.func(gene="CD8A", min=2.0)
    assert r["payload"] == {
        "filterByExpression": {"gene": "CD8A", "min": 2.0, "max": None}
    }
    assert r["matched_cells"] > 0
    assert tool.kind == "ui_action"


async def test_filter_by_gene_expression_range(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    r = await tool.func(gene="CD8A", min=1.0, max=5.0)
    assert r["payload"] == {
        "filterByExpression": {"gene": "CD8A", "min": 1.0, "max": 5.0}
    }


async def test_filter_by_gene_expression_no_bounds(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    r = await tool.func(gene="CD8A")
    assert "error" in r
    assert "at least one" in r["error"]


async def test_filter_by_gene_expression_min_gt_max(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    r = await tool.func(gene="CD8A", min=10.0, max=1.0)
    assert "error" in r
    assert "max" in r["error"]


async def test_filter_by_gene_expression_unknown_gene(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    r = await tool.func(gene="NOT_A_GENE", min=0.0)
    assert "error" in r


async def test_filter_by_gene_expression_no_cells_match(fake_zarr):
    tool = filter_by_gene_expression_tool(fake_zarr)
    # CD8A expression in the fake is rng.random*10 + (T-cell boost), so values
    # are roughly in [0, 15]. min=1000 yields zero matches.
    r = await tool.func(gene="CD8A", min=1000.0)
    assert "error" in r
    assert "no cells match" in r["error"]


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


async def test_set_render_controls_both():
    tool = set_render_controls_tool()
    r = await tool.func(point_size=2.5, opacity=0.7)
    assert r == {"payload": {"pointSize": 2.5, "opacity": 0.7}}
    assert tool.kind == "ui_action"


async def test_set_render_controls_point_size_only():
    tool = set_render_controls_tool()
    r = await tool.func(point_size=4.0)
    assert r == {"payload": {"pointSize": 4.0}}


async def test_set_render_controls_opacity_only():
    tool = set_render_controls_tool()
    r = await tool.func(opacity=0.3)
    assert r == {"payload": {"opacity": 0.3}}


async def test_set_render_controls_neither():
    tool = set_render_controls_tool()
    r = await tool.func()
    assert "error" in r


async def test_set_render_controls_negative_point_size():
    tool = set_render_controls_tool()
    r = await tool.func(point_size=-1.0)
    assert "error" in r  # schema's pointSize must be positive


async def test_set_render_controls_opacity_out_of_range():
    tool = set_render_controls_tool()
    r = await tool.func(opacity=1.5)
    assert "error" in r  # schema's opacity must be in [0, 1]


async def test_set_selection_display_mode_dim():
    tool = set_selection_display_mode_tool()
    r = await tool.func(value="dim")
    assert r == {"payload": {"selectionDisplayMode": "dim"}}
    assert tool.kind == "ui_action"


async def test_set_selection_display_mode_hide():
    tool = set_selection_display_mode_tool()
    r = await tool.func(value="hide")
    assert r == {"payload": {"selectionDisplayMode": "hide"}}


async def test_set_selection_display_mode_invalid_value():
    tool = set_selection_display_mode_tool()
    r = await tool.func(value="other")
    assert "error" in r


async def test_set_category_labels_true():
    tool = set_category_labels_tool()
    r = await tool.func(value=True)
    assert r == {"payload": {"showCategoryLabels": True}}
    assert tool.kind == "ui_action"


async def test_set_category_labels_false():
    tool = set_category_labels_tool()
    r = await tool.func(value=False)
    assert r == {"payload": {"showCategoryLabels": False}}


async def test_clear_color_by_emits_null():
    tool = clear_color_by_tool()
    r = await tool.func()
    assert r == {"payload": {"colorBy": None}}
    assert tool.kind == "ui_action"


async def test_clear_render_controls_emits_nulls():
    tool = clear_render_controls_tool()
    r = await tool.func()
    assert r == {"payload": {"pointSize": None, "opacity": None}}
    assert tool.kind == "ui_action"


async def test_remove_summary_obs_column_emits_payload():
    tool = remove_summary_obs_column_tool()
    r = await tool.func(obs_column="cell_type")
    assert r == {"payload": {"removeSummaryObsColumns": ["cell_type"]}}
    assert tool.kind == "ui_action"


async def test_remove_summary_gene_emits_payload():
    tool = remove_summary_gene_tool()
    r = await tool.func(gene="CD8A")
    assert r == {"payload": {"removeSummaryGenes": ["CD8A"]}}
    assert tool.kind == "ui_action"


async def test_clear_summary_emits_both_nulls():
    tool = clear_summary_tool()
    r = await tool.func()
    assert r == {"payload": {"summaryObsColumns": None, "summaryGenes": None}}
    assert tool.kind == "ui_action"


async def test_clear_viewport_emits_null():
    tool = clear_viewport_tool()
    r = await tool.func()
    assert r == {"payload": {"viewport": None}}
    assert tool.kind == "ui_action"


async def test_fit_viewport_to_selection_emits_true():
    tool = fit_viewport_to_selection_tool()
    r = await tool.func()
    assert r == {"payload": {"fitViewportToSelection": True}}
    assert tool.kind == "ui_action"


async def test_set_color_scale_emits_named_scale():
    tool = set_color_scale_tool()
    r = await tool.func(name="plasma")
    assert r == {"payload": {"colorScaleName": "plasma"}}
    assert tool.kind == "ui_action"


async def test_set_color_scale_rejects_unknown():
    tool = set_color_scale_tool()
    # The Literal type annotation enforces this at the schema layer; with a
    # raw value, validate_partial raises ValidationError. The current tool
    # doesn't catch it, so this propagates as an exception (matching the
    # other open-domain tools' behavior — set_embedding, set_color_by_gene).
    import pytest
    from cell_explorer_agent.schema import ValidationError
    with pytest.raises(ValidationError):
        await tool.func(name="rainbow")  # type: ignore[arg-type]
