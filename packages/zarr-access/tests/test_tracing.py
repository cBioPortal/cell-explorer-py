"""Tests for the zarr-access tracing hook (env-toggled aiohttp TraceConfig)."""

import aiohttp

from zarr_access.tracing import build_trace_configs


def test_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("ZARR_ACCESS_TRACE", raising=False)
    assert build_trace_configs() is None


def test_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("ZARR_ACCESS_TRACE", "0")
    assert build_trace_configs() is None


def test_disabled_when_env_empty(monkeypatch):
    monkeypatch.setenv("ZARR_ACCESS_TRACE", "")
    assert build_trace_configs() is None


def test_enabled_returns_aiohttp_trace_config(monkeypatch):
    monkeypatch.setenv("ZARR_ACCESS_TRACE", "1")
    configs = build_trace_configs()
    assert configs is not None
    assert len(configs) == 1
    assert isinstance(configs[0], aiohttp.TraceConfig)


def test_enabled_truthy_variants(monkeypatch):
    for value in ("true", "TRUE", "yes", "on", "1"):
        monkeypatch.setenv("ZARR_ACCESS_TRACE", value)
        assert build_trace_configs() is not None, f"failed for value={value!r}"


def test_jsonl_file_output(monkeypatch, tmp_path):
    """When ZARR_ACCESS_TRACE_FILE is set, _emit writes JSONL there."""
    import json
    from zarr_access import tracing as t

    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ZARR_ACCESS_TRACE", "1")
    monkeypatch.setenv("ZARR_ACCESS_TRACE_FILE", str(trace_file))
    # Reset any previously-opened file
    t._close_file()

    record = {
        "ts": 1234567890.0,
        "method": "GET",
        "url": "https://example.com/X/c/0",
        "path": "/X/c/0",
        "status": 200,
        "bytes": 1024,
        "elapsed_ms": 12.3,
    }
    t._emit(record, "human-readable line that should NOT go to file")

    # Force flush + close so we can read the file
    t._close_file()

    contents = trace_file.read_text()
    lines = [ln for ln in contents.splitlines() if ln]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == record


def test_no_file_falls_back_to_stderr(monkeypatch, capsys):
    """Without ZARR_ACCESS_TRACE_FILE, _emit writes the human line to stderr."""
    from zarr_access import tracing as t

    monkeypatch.setenv("ZARR_ACCESS_TRACE", "1")
    monkeypatch.delenv("ZARR_ACCESS_TRACE_FILE", raising=False)
    t._close_file()

    t._emit({"foo": "bar"}, "[zarr] GET /x → 200 (1KB, 5ms)")

    captured = capsys.readouterr()
    assert "[zarr] GET /x → 200" in captured.err
