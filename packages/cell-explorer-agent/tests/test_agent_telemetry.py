import pytest

from cell_explorer_agent.agent import ChatAgent
from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import Done
from cell_explorer_agent.llm.base import LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from cell_explorer_agent.messages import UserMessage
from cell_explorer_agent.prompt.dataset_context import DatasetContext, ObsColumnInfo
from cell_explorer_agent.telemetry.fake import FakeLangfuseClient
from cell_explorer_agent.tools.registry import ToolCatalog


class _StubLLM:
    """Yields a scripted sequence of LLM events across one or more streams."""

    def __init__(self, scripts: list[list]) -> None:
        self._scripts = scripts
        self._call = 0

    def stream(self, **kwargs):
        async def gen():
            script = self._scripts[self._call]
            self._call += 1
            for ev in script:
                yield ev

        return gen()


def _ctx() -> DatasetContext:
    return DatasetContext(
        slug="pbmc3k",
        name="PBMC 3k",
        description="",
        n_obs=2700,
        n_var=32000,
        obs_columns=[ObsColumnInfo(name="cell_type", dtype="categorical", cardinality=8)],
        embedding_keys=["X_umap"],
    )


async def test_agent_records_a_trace_per_turn(monkeypatch):
    fake = FakeLangfuseClient()
    monkeypatch.setattr(
        "cell_explorer_agent.telemetry.langfuse_client.get", lambda cfg: fake
    )

    usage = LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    llm = _StubLLM([[LLMTextDelta(text="hello"), LLMStop(reason="end_turn", usage=usage)]])
    agent = ChatAgent(
        llm=llm, catalog=ToolCatalog(), dataset_ctx=_ctx(), config=AgentConfig()
    )

    events = []
    async for ev in agent.run(
        messages=[UserMessage(content="hi")],
        view_state=None,
        telemetry_context={
            "user_id": "user-1",
            "thread_id": "thread-abc",
            "dataset_slug": "pbmc3k",
            "is_public": True,
            "environment": "test",
        },
    ):
        events.append(ev)

    assert any(isinstance(e, Done) for e in events)
    assert len(fake.traces) == 1
    t = fake.traces[0]
    assert t.user_id == "user-1"
    assert t.session_id == "thread-abc"
    assert "visibility:public" in t.tags
    assert len(t.generations) == 1
    assert t.generations[0].output == "hello"
    assert t.generations[0].usage_details["input_tokens"] == 10
    # No tool spans this turn.
    assert t.spans == []


async def test_agent_no_telemetry_context_skips_tracing(monkeypatch):
    """When the API caller omits telemetry_context, no trace is recorded."""
    fake = FakeLangfuseClient()
    monkeypatch.setattr(
        "cell_explorer_agent.telemetry.langfuse_client.get", lambda cfg: fake
    )

    usage = LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    llm = _StubLLM([[LLMTextDelta(text="hello"), LLMStop(reason="end_turn", usage=usage)]])
    agent = ChatAgent(
        llm=llm, catalog=ToolCatalog(), dataset_ctx=_ctx(), config=AgentConfig()
    )

    async for _ in agent.run(messages=[UserMessage(content="hi")], view_state=None):
        pass
    assert fake.traces == []


async def test_agent_private_dataset_redacts(monkeypatch):
    fake = FakeLangfuseClient()
    monkeypatch.setattr(
        "cell_explorer_agent.telemetry.langfuse_client.get", lambda cfg: fake
    )

    usage = LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    llm = _StubLLM([[LLMTextDelta(text="hello"), LLMStop(reason="end_turn", usage=usage)]])
    agent = ChatAgent(
        llm=llm, catalog=ToolCatalog(), dataset_ctx=_ctx(), config=AgentConfig()
    )

    async for _ in agent.run(
        messages=[UserMessage(content="sensitive question")],
        view_state={"embedding": "X_umap"},
        telemetry_context={
            "user_id": "user-1",
            "thread_id": "thread-abc",
            "dataset_slug": "private",
            "is_public": False,
            "environment": "test",
        },
    ):
        pass

    t = fake.traces[0]
    assert "visibility:private" in t.tags
    assert t.input == "[redacted]"
    assert any(u.get("output") == "[redacted]" for u in t.updates)
    assert t.metadata["view_state"] == {"_redacted": "view_state"}
    assert t.generations[0].output == "[redacted]"
