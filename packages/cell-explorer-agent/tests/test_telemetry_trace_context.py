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

    assert len(fake.traces) == 1
    t = fake.traces[0]
    assert t.name == "chat-turn"
    assert t.user_id == "user-1"
    assert t.session_id == "thread-abc"
    assert "visibility:public" in t.tags
    assert "dataset:pbmc3k" in t.tags
    assert "model:claude-sonnet-4-6" in t.tags
    assert t.input == "what are the top genes?"
    assert t.metadata["view_state"] == {"embedding": "X_umap"}
    # Output was set explicitly through set_output, which calls trace.update.
    assert any(u.get("output") == "Final assistant text." for u in t.updates)
    assert len(t.generations) == 1
    assert t.generations[0].output == "Here are the top genes."
    assert t.generations[0].usage_details["input_tokens"] == 100
    assert len(t.spans) == 1
    assert t.spans[0].name == "tool:top_expressed_genes"
    assert t.spans[0].input == {"gene": "CD8A"}


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

    t = fake.traces[0]
    assert "visibility:private" in t.tags
    # Content redacted
    assert t.input == "[redacted]"
    assert t.metadata["view_state"] == {"_redacted": "view_state"}
    assert any(u.get("output") == "[redacted]" for u in t.updates)
    # Structural fields preserved
    assert t.generations[0].output == "[redacted]"
    assert t.generations[0].usage_details["input_tokens"] == 100  # structural — kept
    assert t.spans[0].name == "tool:filter_by_ids"                # structural — kept
    assert t.spans[0].input == {"_redacted": "tool_args", "tool": "filter_by_ids"}


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
    # Reached here without raising — no-op behavior holds.


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
    assert fake.traces[0].user_id == "anonymous"


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
    assert fake.traces[0].spans[0].level == "ERROR"
    assert fake.traces[0].spans[0].status_message == "boom"


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
    # No exception escaped the context manager.
