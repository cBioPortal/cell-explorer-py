"""Confirm Langfuse is flushed on FastAPI shutdown."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from cell_explorer_api.config import Settings
from cell_explorer_api.main import create_app


def test_lifespan_flushes_langfuse():
    settings = Settings()
    app = create_app(settings)
    with patch(
        "cell_explorer_agent.telemetry.langfuse_client.flush"
    ) as mock_flush:
        with TestClient(app):
            pass  # Enter lifespan, then exit
    mock_flush.assert_called_once()
