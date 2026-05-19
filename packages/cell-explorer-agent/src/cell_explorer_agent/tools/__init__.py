"""Tool definitions and catalog."""

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.tools.data.compare import compare_groups_tool
from cell_explorer_agent.tools.strata_protocol import StrataAccess
from cell_explorer_agent.tools.data.genes import (
    gene_expression_summary_tool,
    search_genes_tool,
    top_expressed_genes_tool,
)
from cell_explorer_agent.tools.data.obs import (
    cluster_stats_tool,
    describe_obs_column_tool,
)
from cell_explorer_agent.tools.data.schema import get_dataset_schema_tool
from cell_explorer_agent.tools.registry import Tool, ToolCatalog, ToolKind
from cell_explorer_agent.tools.ui_action.color import (
    clear_color_by_tool,
    set_color_by_category_tool,
    set_color_by_gene_tool,
    set_color_scale_tool,
)
from cell_explorer_agent.tools.ui_action.embedding import set_embedding_tool
from cell_explorer_agent.tools.ui_action.filter import (
    clear_filter_tool,
    filter_by_ids_tool,
)
from cell_explorer_agent.tools.ui_action.filter_by_expression import (
    filter_by_gene_expression_tool,
)
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
from cell_explorer_agent.tools.ui_action.summary import (
    add_summary_gene_tool,
    add_summary_obs_column_tool,
    clear_summary_tool,
    remove_summary_gene_tool,
    remove_summary_obs_column_tool,
)
from cell_explorer_agent.tools.ui_action.summary_context import set_summary_context_tool
from cell_explorer_agent.tools.ui_action.viewport import (
    clear_viewport_tool,
    fit_viewport_to_selection_tool,
    set_viewport_tool,
)
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def build_v1_catalog(
    z: ZarrAccess,
    *,
    config: AgentConfig,
    strata: StrataAccess | None = None,
) -> ToolCatalog:
    cat = ToolCatalog()
    lim = config.tool_result_max_bytes

    # data
    cat.register(get_dataset_schema_tool(z, limit_bytes=lim))
    cat.register(describe_obs_column_tool(z, limit_bytes=lim))
    cat.register(cluster_stats_tool(z, limit_bytes=lim))
    cat.register(search_genes_tool(z, limit_bytes=lim))
    cat.register(gene_expression_summary_tool(z, limit_bytes=lim))
    cat.register(
        top_expressed_genes_tool(
            z,
            limit_bytes=lim,
            concurrency=config.gene_scan_concurrency,
            strata=strata,
        )
    )
    cat.register(
        compare_groups_tool(
            z,
            limit_bytes=lim,
            concurrency=config.gene_scan_concurrency,
            strata=strata,
        )
    )

    # ui_action
    cat.register(set_embedding_tool(z))
    cat.register(set_color_by_gene_tool(z))
    cat.register(set_color_by_category_tool(z))
    cat.register(set_color_scale_tool())
    cat.register(clear_color_by_tool())
    cat.register(filter_by_ids_tool(z, filter_ids_max=config.filter_ids_max))
    cat.register(filter_by_gene_expression_tool(z))
    cat.register(set_selection_display_mode_tool())
    cat.register(set_category_labels_tool())
    cat.register(clear_filter_tool())
    cat.register(add_summary_obs_column_tool(z))
    cat.register(add_summary_gene_tool(z))
    cat.register(remove_summary_obs_column_tool())
    cat.register(remove_summary_gene_tool())
    cat.register(clear_summary_tool())

    # Experimental ui_action tools (Plan 2 view-config redesign).
    # Plan 3 will flip the flag default to True.
    if config.experimental_view_tools:
        cat.register(set_viewport_tool())
        cat.register(clear_viewport_tool())
        cat.register(fit_viewport_to_selection_tool())
        cat.register(set_summary_context_tool())
        cat.register(set_gene_label_column_tool(z))
        cat.register(set_render_controls_tool())
        cat.register(clear_render_controls_tool())

    return cat


__all__ = ["Tool", "ToolCatalog", "ToolKind", "build_v1_catalog"]
