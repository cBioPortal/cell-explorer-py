"""Tool definitions and catalog."""

from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.tools.data.compare import compare_groups_tool
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
    set_color_by_category_tool,
    set_color_by_gene_tool,
)
from cell_explorer_agent.tools.ui_action.embedding import set_embedding_tool
from cell_explorer_agent.tools.ui_action.filter import (
    clear_filter_tool,
    filter_by_ids_tool,
)
from cell_explorer_agent.tools.ui_action.summary import (
    add_summary_gene_tool,
    add_summary_obs_column_tool,
)
from cell_explorer_agent.tools.zarr_protocol import ZarrAccess


def build_v1_catalog(z: ZarrAccess, *, config: AgentConfig) -> ToolCatalog:
    cat = ToolCatalog()
    lim = config.tool_result_max_bytes

    # data
    cat.register(get_dataset_schema_tool(z, limit_bytes=lim))
    cat.register(describe_obs_column_tool(z, limit_bytes=lim))
    cat.register(cluster_stats_tool(z, limit_bytes=lim))
    cat.register(search_genes_tool(z, limit_bytes=lim))
    cat.register(gene_expression_summary_tool(z, limit_bytes=lim))
    cat.register(top_expressed_genes_tool(z, limit_bytes=lim))
    cat.register(compare_groups_tool(z, limit_bytes=lim))

    # ui_action
    cat.register(set_embedding_tool(z))
    cat.register(set_color_by_gene_tool(z))
    cat.register(set_color_by_category_tool(z))
    cat.register(filter_by_ids_tool(z, filter_ids_max=config.filter_ids_max))
    cat.register(clear_filter_tool())
    cat.register(add_summary_obs_column_tool(z))
    cat.register(add_summary_gene_tool(z))

    return cat


__all__ = ["Tool", "ToolCatalog", "ToolKind", "build_v1_catalog"]
