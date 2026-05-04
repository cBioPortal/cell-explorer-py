import json

from cell_explorer_agent.events import (
    TextDelta,
    ToolProgress,
    UIAction,
    Error,
    Done,
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
