import pytest

from cell_explorer_agent.llm.fake import FakeLLMClient, Script
from cell_explorer_agent.llm.base import (
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
)


async def test_fake_emits_scripted_events():
    script = Script(
        events=[
            LLMTextDelta(text="hello "),
            LLMTextDelta(text="world"),
            LLMStop(reason="end_turn", usage=LLMUsage(input_tokens=5, output_tokens=2)),
        ]
    )
    fake = FakeLLMClient(scripts=[script])
    events = []
    async for e in fake.stream(
        system="sys", messages=[], tools=[], model="m", max_tokens=100, timeout_s=10
    ):
        events.append(e)
    assert [type(e).__name__ for e in events] == [
        "LLMTextDelta",
        "LLMTextDelta",
        "LLMStop",
    ]


async def test_fake_emits_tool_call():
    script = Script(
        events=[
            LLMToolCall(id="c1", name="get_dataset_schema", args={}),
            LLMStop(reason="tool_use"),
        ]
    )
    fake = FakeLLMClient(scripts=[script])
    events = [
        e
        async for e in fake.stream(
            system="sys", messages=[], tools=[], model="m", max_tokens=100, timeout_s=10
        )
    ]
    assert isinstance(events[0], LLMToolCall)
    assert events[0].name == "get_dataset_schema"


async def test_fake_exhausted_raises():
    fake = FakeLLMClient(scripts=[])
    with pytest.raises(AssertionError, match="no more scripts"):
        async for _ in fake.stream(
            system="", messages=[], tools=[], model="", max_tokens=1, timeout_s=1
        ):
            pass
