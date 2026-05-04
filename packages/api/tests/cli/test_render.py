from cell_explorer_agent.events import (
    Done,
    Error,
    TextDelta,
    TokenUsage,
    ToolProgress,
    UIAction,
)

from cell_explorer_api.cli.render import render_event_line


def test_render_text_delta_bare_text():
    assert render_event_line(TextDelta(text="hello")) == "hello"


def test_render_tool_progress_started_includes_tool_name():
    line = render_event_line(ToolProgress(tool="get_dataset_schema", status="started"))
    assert "get_dataset_schema" in line


def test_render_tool_progress_ok_is_silent_in_non_verbose():
    line = render_event_line(ToolProgress(tool="x", status="ok"), verbose=False)
    assert line == ""


def test_render_tool_progress_ok_is_visible_in_verbose():
    line = render_event_line(ToolProgress(tool="x", status="ok"), verbose=True)
    assert "x" in line


def test_render_tool_progress_error_always_shown():
    line = render_event_line(ToolProgress(tool="x", status="error", summary="boom"))
    assert "boom" in line
    assert "x" in line


def test_render_ui_action_inline_payload():
    line = render_event_line(UIAction(payload={"colorBy": "gene", "gene": "CD8A"}))
    assert "CD8A" in line


def test_render_error_contains_message():
    line = render_event_line(Error(message="oops"))
    assert "oops" in line


def test_render_done_silent_by_default():
    line = render_event_line(Done(usage=TokenUsage(input_tokens=10, output_tokens=5)))
    assert line == ""


def test_render_done_verbose_shows_usage():
    line = render_event_line(
        Done(usage=TokenUsage(input_tokens=10, output_tokens=5)),
        verbose=True,
    )
    assert "10" in line and "5" in line
