from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.tools import build_v1_catalog


async def test_v1_catalog_includes_all_tools(fake_zarr):
    cat = build_v1_catalog(fake_zarr, config=AgentConfig())
    names = {t.name for t in cat.all()}
    assert names == {
        # data
        "get_dataset_schema",
        "describe_obs_column",
        "cluster_stats",
        "search_genes",
        "gene_expression_summary",
        "top_expressed_genes",
        "gene_panel_by_obs",
        "compare_groups",
        "find_markers",
        # ui_action (existing)
        "set_embedding",
        "set_color_by_gene",
        "set_color_by_category",
        "set_color_scale",
        "clear_color_by",
        "filter_by_ids",
        "filter_by_gene_expression",
        "set_selection_display_mode",
        "set_category_labels",
        "clear_filter",
        "add_summary_obs_column",
        "add_summary_gene",
        "remove_summary_obs_column",
        "remove_summary_gene",
        "clear_summary",
        # ui_action (Plan 2 view-config tools, default-on as of Plan 3)
        "set_viewport",
        "clear_viewport",
        "fit_viewport_to_selection",
        "set_summary_context",
        "set_gene_label_column",
        "set_render_controls",
        "clear_render_controls",
    }


async def test_experimental_view_tools_disabled_via_env(fake_zarr, monkeypatch):
    monkeypatch.setenv("CHAT_EXPERIMENTAL_VIEW_TOOLS", "false")
    cfg = AgentConfig()
    assert cfg.experimental_view_tools is False
    catalog = build_v1_catalog(fake_zarr, config=cfg)
    names = [t.name for t in catalog.all()]
    assert "set_viewport" not in names
    assert "set_summary_context" not in names
    assert "set_gene_label_column" not in names
    assert "set_render_controls" not in names


async def test_experimental_view_tools_enabled_registers_new_tools(fake_zarr, monkeypatch):
    monkeypatch.setenv("CHAT_EXPERIMENTAL_VIEW_TOOLS", "true")
    cfg = AgentConfig()
    assert cfg.experimental_view_tools is True
    catalog = build_v1_catalog(fake_zarr, config=cfg)
    names = [t.name for t in catalog.all()]
    assert "set_viewport" in names
    assert "set_summary_context" in names
    assert "set_gene_label_column" in names
    assert "set_render_controls" in names


async def test_existing_tools_present_with_or_without_flag(fake_zarr):
    cfg_off = AgentConfig()
    cat_off = build_v1_catalog(fake_zarr, config=cfg_off)
    names_off = {t.name for t in cat_off.all()}
    # Existing tools always present
    assert "set_embedding" in names_off
    assert "set_color_by_gene" in names_off
    assert "set_color_by_category" in names_off


from tests.fakes.fake_strata import FakeStrataAccess


def test_compare_groups_registered_once_with_strata(fake_zarr):
    from cell_explorer_agent.config import AgentConfig
    from cell_explorer_agent.tools import build_v1_catalog
    strata = FakeStrataAccess.default()
    catalog = build_v1_catalog(fake_zarr, config=AgentConfig(), strata=strata)
    names = [t.name for t in catalog.all()]
    assert names.count("compare_groups") == 1
    assert "compare_groups_via_strata" not in names


def test_compare_groups_registered_once_without_strata(fake_zarr):
    from cell_explorer_agent.config import AgentConfig
    from cell_explorer_agent.tools import build_v1_catalog
    catalog = build_v1_catalog(fake_zarr, config=AgentConfig(), strata=None)
    names = [t.name for t in catalog.all()]
    assert names.count("compare_groups") == 1
    assert "compare_groups_via_strata" not in names


def test_gene_panel_by_obs_in_v1_catalog(fake_zarr):
    from cell_explorer_agent.config import AgentConfig
    from cell_explorer_agent.tools import build_v1_catalog
    cat = build_v1_catalog(fake_zarr, config=AgentConfig())
    names = [t.name for t in cat.all()]
    assert "gene_panel_by_obs" in names
