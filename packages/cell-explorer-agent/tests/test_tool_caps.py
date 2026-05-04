from cell_explorer_agent.tools.caps import cap_json_bytes


def test_under_cap_returns_payload_unchanged():
    payload = {"a": 1, "b": [1, 2, 3]}
    result = cap_json_bytes(payload, limit_bytes=10_000)
    assert result == payload


def test_over_cap_returns_truncation_marker():
    big = {"values": list(range(100_000))}
    result = cap_json_bytes(big, limit_bytes=1024)
    assert result["_truncated"] is True
    assert "limit_bytes" in result
    assert result["limit_bytes"] == 1024
    assert "hint" in result


def test_exact_cap_returns_payload_unchanged():
    payload = {"a": "x" * 10}
    limit = len(__import__("json").dumps(payload).encode("utf-8"))
    result = cap_json_bytes(payload, limit_bytes=limit)
    assert result == payload
