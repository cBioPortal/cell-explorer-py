import json

from cell_explorer_agent.events import (
    TextDelta,
    ToolProgress,
    UIAction,
    Error,
    Done,
    ThreadOpen,
    TokenUsage,
    ChatEvent,
    dump_event,
)


def test_text_delta_shape():
    e = TextDelta(text="hello")
    d = e.model_dump()
    assert d == {"type": "text_delta", "text": "hello"}


def test_tool_progress_started():
    e = ToolProgress(tool="get_dataset_schema", status="started")
    d = e.model_dump()
    assert d["type"] == "tool_progress"
    assert d["status"] == "started"


def test_ui_action_carries_partial_app_config():
    e = UIAction(payload={"colorBy": "gene", "gene": "CD8A"})
    d = e.model_dump()
    assert d["type"] == "ui_action"
    assert d["payload"]["gene"] == "CD8A"


def test_done_usage():
    e = Done(
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=80,
            cache_write_tokens=0,
        )
    )
    d = e.model_dump()
    assert d["type"] == "done"
    assert d["usage"]["input_tokens"] == 100


def test_dump_event_produces_valid_json_line():
    line = dump_event(TextDelta(text="hi"))
    parsed = json.loads(line)
    assert parsed == {"type": "text_delta", "text": "hi"}
    assert "\n" not in line


def test_thread_open_serializes_to_expected_shape():
    ev = ThreadOpen(thread_id="abc-123", title="Show CD8A in T cells")
    payload = json.loads(ev.model_dump_json())
    assert payload == {
        "type": "thread_open",
        "thread_id": "abc-123",
        "title": "Show CD8A in T cells",
    }


def test_thread_open_is_part_of_chat_event_union():
    # mypy/runtime test: the union accepts ThreadOpen
    ev: ChatEvent = ThreadOpen(thread_id="x", title="y")
    assert ev.type == "thread_open"


def test_tool_progress_accepts_chart_field():
    """ToolProgress optionally carries a chart hint on the ok status."""
    ev = ToolProgress(
        tool="top_expressed_genes",
        status="ok",
        chart={"type": "top_genes_bar", "data": {"genes": []}},
    )
    assert ev.chart == {"type": "top_genes_bar", "data": {"genes": []}}
    # Round-trips through JSON.
    serialized = dump_event(ev)
    assert '"chart"' in serialized
    assert '"top_genes_bar"' in serialized


def test_tool_progress_chart_field_optional():
    """Existing call sites without chart still work."""
    ev = ToolProgress(tool="x", status="started")
    assert ev.chart is None
