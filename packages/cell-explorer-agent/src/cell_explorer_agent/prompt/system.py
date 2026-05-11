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
        "  - For 'show me X cells' / 'where are X cells' / 'highlight X' on a categorical "
        "value, use set_color_by_category(category=<column>, highlight=[<X>]) — the "
        "highlight arg makes X render at full opacity while the rest fade to gray. "
        "Reserve filter_by_ids for explicit isolation requests ('filter to', 'only show', "
        "'isolate'); it hides non-matching cells from the canvas entirely."
    )
    lines.append(
        "  - Emit at most one ui-action per turn unless the user explicitly asks for multiple changes."
    )
    lines.append(
        "  - Do not paraphrase cell barcodes or obs identifiers into free-form text; "
        "use them only as arguments to filter_by_ids."
    )
    lines.append(
        "  - Do not pan/zoom (set_viewport) or change rendering (set_render_controls) "
        "unless the user explicitly asks. The user controls the camera and visual style; "
        "agent-initiated view changes are usually unwelcome."
    )
    lines.append("")
    lines.append("View-control tools:")
    lines.append(
        "  - set_viewport: pan and zoom the scatterplot. target_x/target_y are coordinates "
        "in the embedding's space (NOT pixel coordinates); zoom is log-scale "
        "(0 ≈ fit-to-view, 1 ≈ 2x, 2 ≈ 4x, negative values zoom out). Use only when the "
        "user asks to focus on a specific region or zoom level."
    )
    lines.append(
        "  - set_summary_context: switch the summary panel between 'overall' (statistics "
        "across all cells) and 'selections' (statistics restricted to the current filter). "
        "Useful when comparing a selection's stats against the dataset average."
    )
    lines.append(
        "  - set_gene_label_column: switch the var dataframe column used to display gene "
        "labels (e.g. between Ensembl IDs and HGNC symbols). Use when the user asks about "
        "label format or when the current column is not human-readable."
    )
    lines.append(
        "  - set_render_controls: adjust point_size (pixels, must be > 0; frontend default "
        "0.5) and/or opacity (0=transparent, 1=opaque). At least one parameter required. "
        "Only use when the user explicitly asks to change the visual style."
    )
    return "\n".join(lines)
