from cell_explorer_agent.telemetry.redaction import (
    redact_assistant_output,
    redact_tool_args,
    redact_tool_result,
    redact_user_input,
    redact_view_state,
    REDACTED_SENTINEL,
)


# ---- user input ----

def test_redact_user_input_public_passes_through():
    assert redact_user_input("hello", public=True) == "hello"


def test_redact_user_input_private_returns_sentinel():
    assert redact_user_input("hello", public=False) == REDACTED_SENTINEL


def test_redact_user_input_empty_public():
    assert redact_user_input("", public=True) == ""


# ---- view state ----

def test_redact_view_state_public_passes_through():
    snap = {"embedding": "X_umap", "gene": "CD8A"}
    assert redact_view_state(snap, public=True) == snap


def test_redact_view_state_private_replaces_dict():
    snap = {"embedding": "X_umap", "gene": "CD8A"}
    result = redact_view_state(snap, public=False)
    assert result == {"_redacted": "view_state"}


def test_redact_view_state_none_passes_through():
    # None inputs (no view state sent) are not data and don't need redaction.
    assert redact_view_state(None, public=False) is None
    assert redact_view_state(None, public=True) is None


# ---- tool args ----

def test_redact_tool_args_public_passes_through():
    args = {"gene": "CD8A", "n": 20}
    assert redact_tool_args("top_expressed_genes", args, public=True) == args


def test_redact_tool_args_private_replaces_dict():
    args = {"gene": "CD8A", "n": 20}
    result = redact_tool_args("top_expressed_genes", args, public=False)
    assert result == {"_redacted": "tool_args", "tool": "top_expressed_genes"}


# ---- tool result ----

def test_redact_tool_result_public_returns_full_under_cap():
    result = {"genes": ["A", "B", "C"]}
    assert redact_tool_result(result, public=True) == result


def test_redact_tool_result_public_truncates_over_cap():
    # 4 KB cap. Construct a large payload.
    big = {"genes": ["G" + str(i) for i in range(2000)]}
    out = redact_tool_result(big, public=True)
    # Truncation marker present.
    assert isinstance(out, dict)
    assert out.get("_truncated") is True
    assert "_size_bytes" in out


def test_redact_tool_result_private_replaces_dict():
    result = {"genes": ["A", "B", "C"]}
    out = redact_tool_result(result, public=False)
    assert out == {"_redacted": "tool_result"}


# ---- assistant output ----

def test_redact_assistant_output_public_passes_through():
    assert redact_assistant_output("Answer is X", public=True) == "Answer is X"


def test_redact_assistant_output_private_returns_sentinel():
    assert redact_assistant_output("Answer is X", public=False) == REDACTED_SENTINEL
