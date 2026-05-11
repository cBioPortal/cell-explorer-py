from cell_explorer_agent.prompt.dataset_context import build_dataset_context
from cell_explorer_agent.prompt.system import build_system_prompt


async def test_dataset_context_from_fake(fake_zarr):
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo Dataset", description="A tiny test dataset"
    )
    assert ctx.slug == "demo"
    assert ctx.n_obs == 100
    assert ctx.n_var == 50
    assert "cell_type" in [c.name for c in ctx.obs_columns]
    assert "X_umap" in ctx.embedding_keys


async def test_system_prompt_contains_dataset_metadata(fake_zarr):
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    assert "demo" in sys
    assert "100" in sys
    assert "cell_type" in sys
    assert "X_umap" in sys
    # safety rules are present
    assert "cell barcodes" in sys or "identifier" in sys


async def test_system_prompt_documents_new_view_tools(fake_zarr):
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    # Each new tool is mentioned by name
    assert "set_viewport" in sys
    assert "set_summary_context" in sys
    assert "set_gene_label_column" in sys
    assert "set_render_controls" in sys


async def test_system_prompt_explains_viewport_coord_space(fake_zarr):
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    # Viewport coords are in embedding space, not pixels — critical for the LLM
    assert "embedding" in sys.lower()
    # Some hint about the coordinate space being non-pixel
    assert "pixel" in sys.lower() or "coordinate space" in sys.lower()


async def test_system_prompt_warns_against_proactive_view_changes(fake_zarr):
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    # The LLM should NOT proactively change view state without user prompting
    sys_lower = sys.lower()
    assert "unless" in sys_lower or "only when" in sys_lower or "do not" in sys_lower


async def test_system_prompt_prefers_color_by_over_filter_for_show_queries(fake_zarr):
    """The agent should reach for set_color_by_category on 'show me X' style queries,
    not filter_by_ids (which hides non-matching cells from the view)."""
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    sys_lower = sys.lower()
    # The policy explicitly names set_color_by_category as the right choice for "show me"
    assert "set_color_by_category" in sys
    # The trigger phrasing for "show me" / "where are" type queries is documented
    assert "show me" in sys_lower or "where are" in sys_lower
    # filter_by_ids is reserved for explicit isolation requests
    assert (
        "isolate" in sys_lower
        or "filter to" in sys_lower
        or "only show" in sys_lower
    )


async def test_system_prompt_mentions_highlight_arg(fake_zarr):
    """The prompt should teach the agent about the highlight= arg so it can
    surface a specific category value at full opacity in one tool call."""
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    # The new arg is named so the LLM knows it exists
    assert "highlight" in sys.lower()


async def test_system_prompt_documents_filter_by_gene_expression(fake_zarr):
    """The prompt should name filter_by_gene_expression and direct the agent
    to call gene_expression_summary first to learn expression ranges."""
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    assert "filter_by_gene_expression" in sys
    # Cross-reference: the prompt should mention gene_expression_summary as
    # the pre-resolution step (same pattern as describe_obs_column for
    # filter_by_ids).
    assert "gene_expression_summary" in sys


async def test_system_prompt_directs_describe_obs_column_before_highlight(fake_zarr):
    """The agent should resolve exact category labels via describe_obs_column
    before passing them to either filter_by_ids or set_color_by_category's
    highlight arg. Otherwise it guesses ('T cell' vs 'T.cell') and fails
    tool-side validation."""
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description="A dataset"
    )
    sys = build_system_prompt(ctx)
    # The describe_obs_column policy now covers both consumers
    assert "describe_obs_column" in sys
    # Both consumer tools are named together
    assert "filter_by_ids" in sys and "set_color_by_category" in sys
