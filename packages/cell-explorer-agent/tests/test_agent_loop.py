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
