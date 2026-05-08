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
