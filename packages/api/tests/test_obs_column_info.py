"""The obs column model shared by the chat context and catalogue responses."""

from cell_explorer_api.schemas.obs import ObsColumnInfo


def test_facet_defaults_to_none():
    # Chat builds these without a facet; the field must be optional so chat's
    # response gains `facet: null` rather than failing validation.
    col = ObsColumnInfo(name="tissue", dtype="categorical", cardinality=19)
    assert col.facet is None
    assert col.values is None


def test_carries_a_facet_when_resolved():
    col = ObsColumnInfo(
        name="organ", dtype="categorical", cardinality=3,
        values=["lung", "liver", "brain"], facet="tissue",
    )
    assert col.facet == "tissue"
    assert col.name == "organ", "name stays the real column, not the canonical key"


def test_chat_route_imports_the_shared_model():
    # Guards the collision this move exists to prevent: one model, one schema name.
    from cell_explorer_api.routes import chat
    assert chat.ObsColumnInfo is ObsColumnInfo
