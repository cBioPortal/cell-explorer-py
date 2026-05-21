import pytest


def test_system_prompt_mentions_chart_awareness():
    """The agent is told to write concise summaries when a chart is present
    in the tool result, instead of re-enumerating every row."""
    from cell_explorer_agent.prompt.dataset_context import DatasetContext, ObsColumnInfo
    from cell_explorer_agent.prompt.system import build_system_prompt
    ctx = DatasetContext(
        slug="x", name="x", description="", n_obs=10, n_var=10,
        obs_columns=[ObsColumnInfo(name="ct", dtype="categorical", cardinality=2)],
        embedding_keys=["X_umap"],
    )
    prompt = build_system_prompt(ctx)
    # Chart-awareness hint should be present.
    assert "chart" in prompt.lower()
    # gene_panel_by_obs is registered in the tool-use policy.
    assert "gene_panel_by_obs" in prompt


def _minimal_ctx(**kwargs):
    from cell_explorer_agent.prompt.dataset_context import DatasetContext, ObsColumnInfo
    return DatasetContext(
        slug="test-ds",
        name="Test Dataset",
        description="A test description.",
        n_obs=1000,
        n_var=200,
        obs_columns=[ObsColumnInfo(name="cell_type", dtype="categorical", cardinality=5)],
        embedding_keys=["X_umap"],
        **kwargs,
    )


def test_build_system_prompt_includes_curator_notes_when_present():
    from cell_explorer_agent.prompt.system import build_system_prompt
    addendum = "There is a subtle but important consideration: cells were sorted on CD45."
    ctx = _minimal_ctx(prompt_addendum=addendum)
    prompt = build_system_prompt(ctx)

    assert "=== Curator notes (authoritative dataset context) ===" in prompt
    assert addendum in prompt
    assert "=== end curator notes ===" in prompt

    # Opening fence appears before the shape line.
    idx_fence = prompt.index("=== Curator notes (authoritative dataset context) ===")
    idx_shape = prompt.index("Shape:")
    assert idx_fence < idx_shape

    # Opening fence appears after description.
    idx_desc = prompt.index("A test description.")
    assert idx_desc < idx_fence


def test_build_system_prompt_omits_curator_notes_when_none():
    from cell_explorer_agent.prompt.system import build_system_prompt
    ctx = _minimal_ctx(prompt_addendum=None)
    prompt = build_system_prompt(ctx)
    assert "Curator notes" not in prompt

    # Byte-identical to prompt built from a context without the field set (default None).
    ctx2 = _minimal_ctx()
    assert prompt == build_system_prompt(ctx2)


@pytest.mark.parametrize("addendum", ["", "   \n  \t  "])
def test_build_system_prompt_omits_curator_notes_when_empty(addendum):
    from cell_explorer_agent.prompt.system import build_system_prompt
    ctx = _minimal_ctx(prompt_addendum=addendum)
    prompt = build_system_prompt(ctx)
    assert "Curator notes" not in prompt
