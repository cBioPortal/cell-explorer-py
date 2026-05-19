"""Verify the chat route passes telemetry_context into the agent.

We exercise the inner _ndjson_event_stream coroutine directly with a stub
agent that captures kwargs — running the full FastAPI route through
TestClient would require auth + DB + dataset fixtures that are not the
point of this test.
"""

import uuid
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_ndjson_event_stream_forwards_telemetry_context(monkeypatch):
    captured: dict = {}

    class _CapturingAgent:
        async def run(self, **kwargs):
            captured.update(kwargs)
            from cell_explorer_agent.events import Done, TokenUsage
            yield Done(usage=TokenUsage())

    from cell_explorer_api.routes.chat import _ndjson_event_stream

    thread_id = uuid.uuid4()
    chunks = []
    async for chunk in _ndjson_event_stream(
        _CapturingAgent(),
        [],  # messages
        None,  # view_state
        engine=MagicMock(),
        thread_id=thread_id,
        thread_title="t",
        telemetry_context={
            "user_id": "user-1",
            "thread_id": str(thread_id),
            "dataset_slug": "pbmc3k",
            "is_public": True,
            "environment": "test",
        },
    ):
        chunks.append(chunk)
    assert captured["telemetry_context"]["user_id"] == "user-1"
    assert captured["telemetry_context"]["dataset_slug"] == "pbmc3k"
    assert captured["telemetry_context"]["is_public"] is True
