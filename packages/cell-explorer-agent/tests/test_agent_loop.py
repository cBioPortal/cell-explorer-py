from cell_explorer_agent.agent import ChatAgent
from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import Done, TextDelta
from cell_explorer_agent.llm.base import LLMStop, LLMTextDelta, LLMUsage
from cell_explorer_agent.llm.fake import FakeLLMClient, Script
from cell_explorer_agent.messages import UserMessage
from cell_explorer_agent.prompt.dataset_context import build_dataset_context
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
