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
