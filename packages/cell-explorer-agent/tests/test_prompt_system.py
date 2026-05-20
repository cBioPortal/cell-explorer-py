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
