import pytest

from cell_explorer_agent.tools.registry import Tool, ToolCatalog


async def _noop(**kwargs):
    return {"ok": True}


def test_tool_construction():
    t = Tool(
        name="demo",
        kind="data",
        description="Demo tool",
        args_schema={"type": "object", "properties": {}},
        func=_noop,
    )
    assert t.name == "demo"
    assert t.kind == "data"


def test_catalog_register_and_lookup():
    cat = ToolCatalog()
    t = Tool(
        name="demo",
        kind="data",
        description="Demo",
        args_schema={"type": "object", "properties": {}},
        func=_noop,
    )
    cat.register(t)
    assert cat.get("demo") is t
    assert cat.get("missing") is None
    assert [x.name for x in cat.all()] == ["demo"]


def test_catalog_rejects_duplicate_name():
    cat = ToolCatalog()
    t = Tool(
        name="demo", kind="data", description="", args_schema={}, func=_noop
    )
    cat.register(t)
    with pytest.raises(ValueError, match="already registered"):
        cat.register(t)


def test_catalog_filters_by_kind():
    cat = ToolCatalog()
    cat.register(Tool(name="d", kind="data", description="", args_schema={}, func=_noop))
    cat.register(Tool(name="u", kind="ui_action", description="", args_schema={}, func=_noop))
    assert [t.name for t in cat.by_kind("data")] == ["d"]
    assert [t.name for t in cat.by_kind("ui_action")] == ["u"]


def test_fake_zarr_satisfies_protocol():
    from cell_explorer_agent.tools.zarr_protocol import ZarrAccess
    from tests.fakes.fake_zarr import FakeZarrAccess

    assert isinstance(FakeZarrAccess.default(), ZarrAccess)
