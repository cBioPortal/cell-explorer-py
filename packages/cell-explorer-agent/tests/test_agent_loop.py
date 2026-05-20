from cell_explorer_agent.agent import ChatAgent
from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import Done, TextDelta, ToolProgress
from cell_explorer_agent.llm.base import LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from cell_explorer_agent.llm.fake import FakeLLMClient, Script
from cell_explorer_agent.messages import UserMessage
from cell_explorer_agent.prompt.dataset_context import build_dataset_context
from cell_explorer_agent.tools.data.schema import get_dataset_schema_tool
from cell_explorer_agent.tools.registry import ToolCatalog


async def test_text_only_turn(fake_zarr):
    llm = FakeLLMClient(
        scripts=[
            Script(
                events=[
                    LLMTextDelta(text="hello "),
                    LLMTextDelta(text="world"),
                    LLMStop(
                        reason="end_turn",
                        usage=LLMUsage(input_tokens=10, output_tokens=3),
                    ),
                ]
            )
        ]
    )
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description=""
    )
    agent = ChatAgent(
        llm=llm,
        catalog=ToolCatalog(),
        dataset_ctx=ctx,
        config=AgentConfig(),
    )
    events = [e async for e in agent.run(messages=[UserMessage(content="hi")])]

    types = [type(e).__name__ for e in events]
    assert types == ["TextDelta", "TextDelta", "Done"]
    assert isinstance(events[-1], Done)
    assert events[-1].usage.input_tokens == 10
    assert events[-1].usage.output_tokens == 3


async def test_data_tool_turn(fake_zarr):
    llm = FakeLLMClient(
        scripts=[
            # First LLM call: model requests the schema tool
            Script(
                events=[
                    LLMToolCall(id="t1", name="get_dataset_schema", args={}),
                    LLMStop(reason="tool_use", usage=LLMUsage(input_tokens=20, output_tokens=5)),
                ]
            ),
            # Second LLM call (after tool result injected): model finishes
            Script(
                events=[
                    LLMTextDelta(text="It has 100 cells and 50 genes."),
                    LLMStop(
                        reason="end_turn",
                        usage=LLMUsage(input_tokens=60, output_tokens=10),
                    ),
                ]
            ),
        ]
    )
    ctx = await build_dataset_context(
        fake_zarr, slug="demo", name="Demo", description=""
    )
    cat = ToolCatalog()
    cat.register(get_dataset_schema_tool(fake_zarr, limit_bytes=32_768))

    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    events = [
        e async for e in agent.run(messages=[UserMessage(content="how big is it?")])
    ]

    kinds = [type(e).__name__ for e in events]
    # tool progress (started, ok), text delta, done
    assert "ToolProgress" in kinds
    assert events[-1].__class__.__name__ == "Done"
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "100 cells" in text

    # Why-panel wire fields: 'started' carries args, 'ok' carries duration_ms,
    # both carry tool_call_id (used by citation back-links).
    progress = [e for e in events if isinstance(e, ToolProgress)]
    started = next(e for e in progress if e.status == "started")
    finished = next(e for e in progress if e.status == "ok")
    assert started.args == {}
    assert started.duration_ms is None
    assert finished.duration_ms is not None
    assert finished.duration_ms >= 0
    assert finished.args is None
    assert started.tool_call_id == "t1"
    assert finished.tool_call_id == "t1"


from cell_explorer_agent.events import UIAction
from cell_explorer_agent.tools.ui_action.color import set_color_by_gene_tool


async def test_ui_action_turn(fake_zarr):
    llm = FakeLLMClient(
        scripts=[
            Script(
                events=[
                    LLMToolCall(
                        id="u1",
                        name="set_color_by_gene",
                        args={"gene": "CD8A"},
                    ),
                    LLMStop(reason="tool_use", usage=LLMUsage()),
                ]
            ),
            Script(
                events=[
                    LLMTextDelta(text="Coloring by CD8A."),
                    LLMStop(reason="end_turn", usage=LLMUsage(output_tokens=4)),
                ]
            ),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="demo", name="Demo", description="")
    cat = ToolCatalog()
    cat.register(set_color_by_gene_tool(fake_zarr))

    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    events = [
        e async for e in agent.run(messages=[UserMessage(content="color by CD8A")])
    ]

    actions = [e for e in events if isinstance(e, UIAction)]
    assert len(actions) == 1
    assert actions[0].payload == {"colorBy": "gene", "gene": "CD8A"}
    assert events[-1].__class__.__name__ == "Done"


async def test_ui_action_validation_failure_never_streams(fake_zarr):
    llm = FakeLLMClient(
        scripts=[
            Script(
                events=[
                    LLMToolCall(
                        id="u1",
                        name="set_color_by_gene",
                        args={"gene": "NOT_A_GENE"},
                    ),
                    LLMStop(reason="tool_use", usage=LLMUsage()),
                ]
            ),
            Script(
                events=[
                    LLMTextDelta(text="That gene isn't in the dataset."),
                    LLMStop(reason="end_turn", usage=LLMUsage()),
                ]
            ),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="demo", name="Demo", description="")
    cat = ToolCatalog()
    cat.register(set_color_by_gene_tool(fake_zarr))

    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    events = [
        e async for e in agent.run(messages=[UserMessage(content="color by NOT_A_GENE")])
    ]

    assert not any(isinstance(e, UIAction) for e in events)


import asyncio

import pytest

from cell_explorer_agent.events import Error
from cell_explorer_agent.tools.registry import Tool


async def _raising_tool(**_):
    raise RuntimeError("boom")


async def test_tool_exception_becomes_tool_result(fake_zarr):
    llm = FakeLLMClient(
        scripts=[
            Script(
                events=[
                    LLMToolCall(id="t1", name="explode", args={}),
                    LLMStop(reason="tool_use", usage=LLMUsage()),
                ]
            ),
            Script(
                events=[
                    LLMTextDelta(text="Sorry, that failed."),
                    LLMStop(reason="end_turn", usage=LLMUsage()),
                ]
            ),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(
        Tool(
            name="explode",
            kind="data",
            description="",
            args_schema={"type": "object", "properties": {}},
            func=_raising_tool,
        )
    )
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    events = [e async for e in agent.run(messages=[UserMessage(content="x")])]

    progress_errors = [
        e for e in events if isinstance(e, ToolProgress) and e.status == "error"
    ]
    assert len(progress_errors) == 1
    assert events[-1].__class__.__name__ == "Done"


async def test_tool_call_start_end_log_lines(fake_zarr, caplog):
    """Every successful tool call emits start + end log lines with tool name,
    kind, and duration so docker logs can show real-time agent activity."""
    llm = FakeLLMClient(
        scripts=[
            Script(events=[
                LLMToolCall(id="t1", name="get_dataset_schema", args={}),
                LLMStop(reason="tool_use", usage=LLMUsage()),
            ]),
            Script(events=[
                LLMTextDelta(text="ok"),
                LLMStop(reason="end_turn", usage=LLMUsage()),
            ]),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(get_dataset_schema_tool(fake_zarr, limit_bytes=32_768))
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())

    with caplog.at_level("INFO", logger="cell_explorer_agent.agent"):
        _ = [e async for e in agent.run(messages=[UserMessage(content="x")])]

    msgs = [r.getMessage() for r in caplog.records]
    starts = [m for m in msgs if m.startswith("chat.tool.call.start")]
    ends = [m for m in msgs if m.startswith("chat.tool.call.end")]
    assert len(starts) == 1, msgs
    assert "tool=get_dataset_schema" in starts[0]
    assert "kind=data" in starts[0]
    assert len(ends) == 1, msgs
    assert "tool=get_dataset_schema" in ends[0]
    assert "duration_ms=" in ends[0]


async def test_tool_call_error_log_includes_duration(fake_zarr, caplog):
    llm = FakeLLMClient(
        scripts=[
            Script(events=[
                LLMToolCall(id="t1", name="explode", args={}),
                LLMStop(reason="tool_use", usage=LLMUsage()),
            ]),
            Script(events=[
                LLMTextDelta(text="oops"),
                LLMStop(reason="end_turn", usage=LLMUsage()),
            ]),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(Tool(
        name="explode", kind="data", description="",
        args_schema={"type": "object", "properties": {}}, func=_raising_tool,
    ))
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())

    with caplog.at_level("ERROR", logger="cell_explorer_agent.agent"):
        _ = [e async for e in agent.run(messages=[UserMessage(content="x")])]

    errs = [r.getMessage() for r in caplog.records if "chat.tool.call.error" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in caplog.records]
    assert "tool=explode" in errs[0]
    assert "duration_ms=" in errs[0]
    assert "correlation=" in errs[0]


async def test_max_tool_calls_cap(fake_zarr):
    # LLM keeps calling a tool forever; cap should kick in.
    def make_repeater(id_prefix):
        return Script(
            events=[
                LLMToolCall(id=f"{id_prefix}-1", name="get_dataset_schema", args={}),
                LLMStop(reason="tool_use", usage=LLMUsage()),
            ]
        )

    scripts = [make_repeater(f"s{i}") for i in range(20)]
    llm = FakeLLMClient(scripts=scripts)
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(get_dataset_schema_tool(fake_zarr, limit_bytes=32_768))

    cfg = AgentConfig()
    # Force the cap
    cfg = AgentConfig(**{**cfg.model_dump(), "max_tool_calls_per_turn": 3})

    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=cfg)
    events = [e async for e in agent.run(messages=[UserMessage(content="loop")])]

    # Final Error + Done
    errors = [e for e in events if isinstance(e, Error)]
    assert len(errors) == 1
    assert "cap" in errors[0].message.lower() or "limit" in errors[0].message.lower()
    assert events[-1].__class__.__name__ == "Done"


async def test_cancellation_propagates(fake_zarr):
    async def slow_tool(**_):
        await asyncio.sleep(1.0)
        return {"ok": True}

    llm = FakeLLMClient(
        scripts=[
            Script(
                events=[
                    LLMToolCall(id="t1", name="slow", args={}),
                    LLMStop(reason="tool_use", usage=LLMUsage()),
                ]
            )
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(
        Tool(
            name="slow",
            kind="data",
            description="",
            args_schema={"type": "object", "properties": {}},
            func=slow_tool,
        )
    )
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())

    async def consume():
        async for _ in agent.run(messages=[UserMessage(content="x")]):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_run_forwards_view_state_block_to_llm(fake_zarr):
    """Non-empty view_state → formatted block reaches the LLM client."""
    script = Script(
        events=[
            LLMTextDelta(text="ok"),
            LLMStop(reason="end_turn", usage=LLMUsage()),
        ]
    )
    llm = FakeLLMClient(scripts=[script])
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    agent = ChatAgent(llm=llm, catalog=ToolCatalog(), dataset_ctx=ctx, config=AgentConfig())
    async for _ in agent.run(
        messages=[UserMessage(content="zoom in more")],
        view_state={"embedding": "X_umap", "gene": "CD8A"},
    ):
        pass
    assert script.called_with_view_state_block is not None
    assert "X_umap" in script.called_with_view_state_block
    assert "CD8A" in script.called_with_view_state_block


async def test_run_omits_view_state_block_when_empty(fake_zarr):
    """Empty / missing view_state → no block sent (default view)."""
    script = Script(
        events=[
            LLMTextDelta(text="ok"),
            LLMStop(reason="end_turn", usage=LLMUsage()),
        ]
    )
    llm = FakeLLMClient(scripts=[script])
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    agent = ChatAgent(llm=llm, catalog=ToolCatalog(), dataset_ctx=ctx, config=AgentConfig())
    async for _ in agent.run(
        messages=[UserMessage(content="hi")],
        view_state={},
    ):
        pass
    assert script.called_with_view_state_block is None


async def test_run_view_state_block_sent_on_every_iteration(fake_zarr):
    """The view-state block must reach the LLM on each iteration of a multi-tool turn,
    not just the first — the agent's internal tool round-trips re-call llm.stream()."""
    s1 = Script(
        events=[
            LLMToolCall(id="t1", name="get_dataset_schema", args={}),
            LLMStop(reason="tool_use", usage=LLMUsage()),
        ]
    )
    s2 = Script(
        events=[
            LLMTextDelta(text="done"),
            LLMStop(reason="end_turn", usage=LLMUsage()),
        ]
    )
    llm = FakeLLMClient(scripts=[s1, s2])
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(get_dataset_schema_tool(fake_zarr, limit_bytes=32_768))
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    async for _ in agent.run(
        messages=[UserMessage(content="hi")],
        view_state={"embedding": "X_umap"},
    ):
        pass
    # Both LLM round-trips should have received the same view-state block
    assert s1.called_with_view_state_block is not None
    assert "X_umap" in s1.called_with_view_state_block
    assert s2.called_with_view_state_block == s1.called_with_view_state_block


async def test_tool_with_wants_context_receives_view_state(fake_zarr):
    """A tool that declares wants_context=True must receive the current
    view_state via the private _context kwarg at dispatch time."""
    captured: dict[str, Any] = {}

    async def _peek_context(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    llm = FakeLLMClient(
        scripts=[
            Script(events=[
                LLMToolCall(id="t1", name="peek", args={"x": 1}),
                LLMStop(reason="tool_use", usage=LLMUsage()),
            ]),
            Script(events=[
                LLMTextDelta(text="ok"),
                LLMStop(reason="end_turn", usage=LLMUsage()),
            ]),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(Tool(
        name="peek", kind="data", description="",
        args_schema={
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        },
        func=_peek_context,
        wants_context=True,
    ))
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    async for _ in agent.run(
        messages=[UserMessage(content="hi")],
        view_state={"colorMode": "category", "category": "leiden"},
    ):
        pass

    assert captured.get("x") == 1
    assert captured.get("_context") == {
        "view_state": {"colorMode": "category", "category": "leiden"}
    }


async def test_tool_without_wants_context_does_not_get_context(fake_zarr):
    """The default (wants_context=False) means _context is NOT injected —
    tools must opt in explicitly."""
    captured: dict[str, Any] = {}

    async def _peek_no_context(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    llm = FakeLLMClient(
        scripts=[
            Script(events=[
                LLMToolCall(id="t1", name="peek", args={}),
                LLMStop(reason="tool_use", usage=LLMUsage()),
            ]),
            Script(events=[
                LLMTextDelta(text="ok"),
                LLMStop(reason="end_turn", usage=LLMUsage()),
            ]),
        ]
    )
    ctx = await build_dataset_context(fake_zarr, slug="d", name="D", description="")
    cat = ToolCatalog()
    cat.register(Tool(
        name="peek", kind="data", description="",
        args_schema={"type": "object", "properties": {}},
        func=_peek_no_context,
    ))
    agent = ChatAgent(llm=llm, catalog=cat, dataset_ctx=ctx, config=AgentConfig())
    async for _ in agent.run(
        messages=[UserMessage(content="hi")],
        view_state={"colorMode": "gene"},
    ):
        pass

    assert "_context" not in captured


def test_strip_chart_data_for_llm():
    """The helper that prepares LLM-side tool result content strips chart.data."""
    from cell_explorer_agent.agent import _strip_chart_data_for_llm

    result = {
        "data": {"foo": "bar"},
        "chart": {"type": "demo", "data": {"foo": "bar"}},
    }
    stripped = _strip_chart_data_for_llm(result)
    assert stripped["data"] == {"foo": "bar"}
    assert stripped["chart"] == {"type": "demo"}
    # Original input not mutated.
    assert result["chart"]["data"] == {"foo": "bar"}


def test_strip_chart_data_passes_through_when_no_chart():
    from cell_explorer_agent.agent import _strip_chart_data_for_llm
    result = {"data": {"x": 1}}
    assert _strip_chart_data_for_llm(result) == {"data": {"x": 1}}


def test_strip_chart_data_passes_through_when_chart_has_no_data():
    from cell_explorer_agent.agent import _strip_chart_data_for_llm
    result = {"data": {"x": 1}, "chart": {"type": "demo"}}
    stripped = _strip_chart_data_for_llm(result)
    assert stripped["chart"] == {"type": "demo"}


def test_strip_chart_data_handles_non_dict_chart():
    """Defensive — if chart is somehow not a dict, pass through."""
    from cell_explorer_agent.agent import _strip_chart_data_for_llm
    result = {"data": {"x": 1}, "chart": "not a dict"}
    # Should not raise; original returned unchanged.
    assert _strip_chart_data_for_llm(result) == result
