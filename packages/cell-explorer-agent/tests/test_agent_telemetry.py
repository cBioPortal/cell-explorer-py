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
    assert fake.root_observation is not None
    assert fake.trace_field("user_id") == "user-1"
    assert fake.trace_field("session_id") == "thread-abc"
    assert "visibility:public" in fake.trace_field("tags")
    assert len(fake.generations) == 1
    assert fake.generations[0].output == "hello"
    assert fake.generations[0].usage_details["input_tokens"] == 10
    # No tool spans this turn.
    assert fake.tool_spans == []


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
    assert fake.observations == [] and fake.trace_updates == []


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

    assert "visibility:private" in fake.trace_field("tags")
    assert fake.root_observation.input == "[redacted]"
    assert fake.trace_field("output") == "[redacted]"
    assert fake.root_observation.metadata["view_state"] == {"_redacted": "view_state"}
    assert fake.generations[0].output == "[redacted]"


async def test_agent_private_dataset_ships_content_when_redaction_off(monkeypatch):
    """LANGFUSE_REDACT_PRIVATE=false (self-hosted Langfuse): private content is
    sent, but the trace is still tagged visibility:private."""
    fake = FakeLangfuseClient()
    monkeypatch.setattr(
        "cell_explorer_agent.telemetry.langfuse_client.get", lambda cfg: fake
    )
    monkeypatch.setenv("LANGFUSE_REDACT_PRIVATE", "false")

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

    tags = fake.trace_field("tags")
    assert "visibility:private" in tags
    assert "visibility:public" not in tags
    assert fake.root_observation.input == "sensitive question"
    assert fake.trace_field("output") == "hello"
    assert fake.root_observation.metadata["view_state"] == {"embedding": "X_umap"}
    assert fake.generations[0].output == "hello"


@pytest.mark.parametrize("redact_private", [True, False])
async def test_agent_public_dataset_unaffected_by_redaction_toggle(monkeypatch, redact_private):
    fake = FakeLangfuseClient()
    monkeypatch.setattr(
        "cell_explorer_agent.telemetry.langfuse_client.get", lambda cfg: fake
    )

    usage = LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0)
    llm = _StubLLM([[LLMTextDelta(text="hello"), LLMStop(reason="end_turn", usage=usage)]])
    agent = ChatAgent(
        llm=llm,
        catalog=ToolCatalog(),
        dataset_ctx=_ctx(),
        config=AgentConfig(langfuse_redact_private=redact_private),
    )

    async for _ in agent.run(
        messages=[UserMessage(content="what are the top genes?")],
        view_state={"embedding": "X_umap"},
        telemetry_context={
            "user_id": "user-1",
            "thread_id": "thread-abc",
            "dataset_slug": "pbmc3k",
            "is_public": True,
            "environment": "test",
        },
    ):
        pass

    assert "visibility:public" in fake.trace_field("tags")
    assert fake.root_observation.input == "what are the top genes?"
    assert fake.trace_field("output") == "hello"
    assert fake.generations[0].output == "hello"
