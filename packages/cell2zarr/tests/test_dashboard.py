"""Tests for the conversion run dashboard."""
import json
import os
import importlib
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def dashboard_client(tmp_path):
    """Create a test client with sample run data."""
    runs_file = tmp_path / "conversion-runs.json"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Write a sample run entry
    runs = [
        {
            "run": 1,
            "date": "2026-03-31",
            "status": "completed",
            "log_file": "logs/run-1.log",
            "dataset": {"n_obs": 1000, "n_vars": 500},
            "performance": {
                "start_time": "2026-03-31T10:00:00Z",
                "end_time": "2026-03-31T10:05:00Z",
                "total_time_s": 300,
            },
        },
        {
            "run": 2,
            "date": "2026-03-31",
            "status": "running",
            "log_file": "logs/run-2.log",
        },
    ]
    runs_file.write_text(json.dumps(runs))

    # Write sample log files
    (logs_dir / "run-1.log").write_text(
        "[2026-03-31 10:00:00] INFO: Phase 1 complete\n"
        "[2026-03-31 10:05:00] INFO: Done\n"
    )
    (logs_dir / "run-2.log").write_text(
        "[2026-03-31 10:10:00] INFO: Phase 1 starting\n"
    )

    # Set env var and reload the dashboard module
    os.environ["CELL2ZARR_RUN_DB"] = str(runs_file)
    import cell2zarr.dashboard.main as dashboard_module
    importlib.reload(dashboard_module)

    yield TestClient(dashboard_module.app)

    # Cleanup
    del os.environ["CELL2ZARR_RUN_DB"]


class TestDashboardIndex:
    def test_returns_200(self, dashboard_client):
        resp = dashboard_client.get("/")
        assert resp.status_code == 200

    def test_contains_run_info(self, dashboard_client):
        resp = dashboard_client.get("/")
        assert "Run 1" in resp.text
        assert "Run 2" in resp.text


class TestApiRuns:
    def test_returns_runs(self, dashboard_client):
        resp = dashboard_client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["run"] == 1
        assert data[1]["run"] == 2

    def test_run_fields(self, dashboard_client):
        resp = dashboard_client.get("/api/runs")
        run = resp.json()[0]
        assert run["status"] == "completed"
        assert run["dataset"]["n_obs"] == 1000


class TestApiLogs:
    def test_returns_log_content(self, dashboard_client):
        resp = dashboard_client.get("/api/logs/1")
        assert resp.status_code == 200
        assert "Phase 1 complete" in resp.text

    def test_missing_run_returns_404(self, dashboard_client):
        resp = dashboard_client.get("/api/logs/999")
        assert resp.status_code == 404

    def test_no_cache_header(self, dashboard_client):
        resp = dashboard_client.get("/api/logs/1")
        assert resp.headers["cache-control"] == "no-store"


class TestApiLogsStream:
    def test_completed_run_streams_content_and_stops(self, dashboard_client):
        """Completed runs stream all content then send done event and close."""
        with dashboard_client.stream("GET", "/api/logs/1/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = list(resp.iter_lines())
        assert any("Phase 1 complete" in l for l in lines)
        assert any("done" in l for l in lines)

    def test_missing_run_returns_404(self, dashboard_client):
        resp = dashboard_client.get("/api/logs/999/stream")
        assert resp.status_code == 404
