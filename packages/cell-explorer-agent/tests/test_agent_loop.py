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
