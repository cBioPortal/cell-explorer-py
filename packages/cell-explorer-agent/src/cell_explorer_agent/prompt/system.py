"""System prompt builder.

The prompt's dataset-metadata block is stable for a chat session, so downstream
adapters can mark it as a cache prefix (Anthropic prompt caching).
"""

from cell_explorer_agent.prompt.dataset_context import DatasetContext


def build_system_prompt(ctx: DatasetContext) -> str:
    lines: list[str] = []
    lines.append(
        "You are a data-exploration assistant for a single single-cell RNA-seq dataset."
    )
    lines.append(
        "Answer user questions about the dataset. Use tools to fetch facts; never invent numbers."
    )
    lines.append("")
    lines.append(f"Dataset: {ctx.slug} — {ctx.name}")
    if ctx.description:
        lines.append(f"Description: {ctx.description}")
    lines.append(f"Shape: {ctx.n_obs} cells × {ctx.n_var} genes.")
    lines.append("")
    lines.append("Obs columns:")
    for c in ctx.obs_columns:
        if c.dtype == "categorical" and c.cardinality is not None:
            lines.append(f"  - {c.name} (categorical, {c.cardinality} categories)")
        else:
            lines.append(f"  - {c.name} ({c.dtype})")
    lines.append("")
    lines.append(f"Available embeddings: {', '.join(ctx.embedding_keys)}")
    lines.append("")
    lines.append("Tool-use policy:")
    lines.append(
        "  - Prefer bounded reads (get_dataset_schema, describe_obs_column) before heavy queries."
    )
    lines.append(
        "  - Use describe_obs_column to learn category values before calling filter_by_ids."
    )
    lines.append(
        "  - Emit at most one ui-action per turn unless the user explicitly asks for multiple changes."
    )
    lines.append(
        "  - Do not paraphrase cell barcodes or obs identifiers into free-form text; "
        "use them only as arguments to filter_by_ids."
    )
    return "\n".join(lines)
