import pytest

from cell_explorer_agent.telemetry.fake import FakeLangfuseClient
from cell_explorer_agent.telemetry.trace_context import TurnTrace


@pytest.fixture
def fake() -> FakeLangfuseClient:
    return FakeLangfuseClient()


async def test_public_dataset_trace_records_full_content(fake):
    async with TurnTrace(
        client=fake,
        user_id="user-1",
        thread_id="thread-abc",
        dataset_slug="pbmc3k",
        is_public=True,
        model="claude-sonnet-4-6",
        environment="test",
        user_input="what are the top genes?",
        view_state={"embedding": "X_umap"},
    ) as trace:
        trace.add_generation(
            input_messages=[{"role": "user", "content": "what are the top genes?"}],
            output_text="Here are the top genes.",
            usage={"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        trace.add_tool_span(
            name="top_expressed_genes",
            args={"gene": "CD8A"},
            result={"genes": ["A", "B"]},
            duration_ms=42,
            status="ok",
        )
        trace.set_output("Final assistant text.")

    # Root observation captured the user input.
    root = fake.root_observation
    assert root is not None
    assert root.name == "chat-turn"
    assert root.as_type == "span"
    assert root.input == "what are the top genes?"
    assert root.metadata["view_state"] == {"embedding": "X_umap"}

    # Trace-level fields were set via update_current_trace.
    assert fake.trace_field("user_id") == "user-1"
    assert fake.trace_field("session_id") == "thread-abc"
    tags = fake.trace_field("tags")
    assert "visibility:public" in tags
    assert "dataset:pbmc3k" in tags
    assert "model:claude-sonnet-4-6" in tags

    # Generation captured the LLM call.
    assert len(fake.generations) == 1
    gen = fake.generations[0]
    assert gen.output == "Here are the top genes."
    assert gen.usage_details["input_tokens"] == 100
    assert gen.ended

    # Tool span captured the dispatch.
    assert len(fake.tool_spans) == 1
    sp = fake.tool_spans[0]
    assert sp.name == "tool:top_expressed_genes"
    assert sp.input == {"gene": "CD8A"}
    assert sp.ended

    # Trace-level output was set via update_current_trace.
    assert fake.trace_field("output") == "Final assistant text."


async def test_private_dataset_trace_redacts_content(fake):
    async with TurnTrace(
        client=fake,
        user_id="user-1",
        thread_id="thread-abc",
        dataset_slug="private-ovarian",
        is_public=False,
        model="claude-sonnet-4-6",
        environment="test",
        user_input="sensitive question",
        view_state={"embedding": "X_umap"},
    ) as trace:
        trace.add_generation(
            input_messages=[{"role": "user", "content": "sensitive question"}],
            output_text="Sensitive answer.",
            usage={"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        trace.add_tool_span(
            name="filter_by_ids",
            args={"ids": ["barcode1", "barcode2"]},
            result={"matched": 2},
            duration_ms=10,
            status="ok",
        )
        trace.set_output("Sensitive final text.")

    tags = fake.trace_field("tags")
    assert "visibility:private" in tags

    # Content redacted on root and trace.
    assert fake.root_observation.input == "[redacted]"
    assert fake.root_observation.metadata["view_state"] == {"_redacted": "view_state"}
    assert fake.trace_field("output") == "[redacted]"

    # Structural fields preserved.
    assert fake.generations[0].output == "[redacted]"
    assert fake.generations[0].usage_details["input_tokens"] == 100
    assert fake.tool_spans[0].name == "tool:filter_by_ids"
    assert fake.tool_spans[0].input == {"_redacted": "tool_args", "tool": "filter_by_ids"}


async def test_no_client_is_noop():
    """When client=None, the context manager records nothing and does not raise."""
    async with TurnTrace(
        client=None,
        user_id="user-1",
        thread_id="thread-abc",
        dataset_slug="x",
        is_public=True,
        model="m",
        environment="test",
        user_input="hi",
        view_state=None,
    ) as trace:
        trace.add_generation(
            input_messages=[],
            output_text="",
            usage={"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0},
        )
        trace.add_tool_span(name="x", args={}, result={}, duration_ms=0, status="ok")
        trace.set_output("done")


async def test_anonymous_user_id_fallback(fake):
    async with TurnTrace(
        client=fake,
        user_id=None,
        thread_id="thread-abc",
        dataset_slug="pbmc3k",
        is_public=True,
        model="m",
        environment="test",
        user_input="hi",
        view_state=None,
    ) as trace:
        trace.set_output("done")
    assert fake.trace_field("user_id") == "anonymous"


async def test_tool_span_error_sets_level_error(fake):
    async with TurnTrace(
        client=fake,
        user_id="user-1",
        thread_id="thread-abc",
        dataset_slug="pbmc3k",
        is_public=True,
        model="m",
        environment="test",
        user_input="hi",
        view_state=None,
    ) as trace:
        trace.add_tool_span(
            name="failing_tool",
            args={},
            result={"error": "boom"},
            duration_ms=5,
            status="error",
        )
    sp = fake.tool_spans[0]
    assert sp.level == "ERROR"
    assert sp.status_message == "boom"


async def test_exception_during_use_does_not_propagate_through_telemetry(fake):
    """If add_generation hits a bug, chat shouldn't fail."""
    async with TurnTrace(
        client=fake,
        user_id="user-1",
        thread_id="thread-abc",
        dataset_slug="pbmc3k",
        is_public=True,
        model="m",
        environment="test",
        user_input="hi",
        view_state=None,
    ) as trace:
        # Force a bug: pass usage of wrong type. The method should swallow.
        trace.add_generation(
            input_messages=[],
            output_text="x",
            usage="not a dict",  # type: ignore[arg-type]
        )
        trace.set_output("done")
