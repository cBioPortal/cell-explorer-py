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
        "compare_groups",
        # ui_action
        "set_embedding",
        "set_color_by_gene",
        "set_color_by_category",
        "filter_by_ids",
        "clear_filter",
        "add_summary_obs_column",
        "add_summary_gene",
    }
