"""Tests for progressive run log writing."""
import json
import pytest

from cell2zarr.run_log import read_runs, write_runs, get_next_run_number, update_run_entry
from cell2zarr.models import RunEntry, RunDataset, RunPerformance, RunZarrConfig


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "conversion-runs.json"


class TestReadWriteRuns:
    def test_read_missing_file(self, log_path):
        assert read_runs(log_path) == []

    def test_read_empty_file(self, log_path):
        log_path.write_text("")
        assert read_runs(log_path) == []

    def test_roundtrip(self, log_path):
        runs = [RunEntry(run=1, status="completed")]
        write_runs(log_path, runs)
        assert read_runs(log_path) == runs

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "runs.json"
        write_runs(nested, [RunEntry(run=1)])
        assert read_runs(nested) == [RunEntry(run=1)]


class TestGetNextRunNumber:
    def test_empty_list(self):
        assert get_next_run_number([]) == 1

    def test_sequential(self):
        assert get_next_run_number([RunEntry(run=1), RunEntry(run=2)]) == 3

    def test_gaps(self):
        assert get_next_run_number([RunEntry(run=1), RunEntry(run=5)]) == 6


class TestUpdateRunEntry:
    def test_updates_scalar_field(self, log_path):
        write_runs(log_path, [RunEntry(run=1, status="running")])
        update_run_entry(log_path, 1, {"status": "completed"})
        runs = read_runs(log_path)
        assert runs[0].status == "completed"

    def test_merges_dict_field(self, log_path):
        write_runs(log_path, [RunEntry(run=1, performance=RunPerformance(start_time="t0"))])
        update_run_entry(log_path, 1, {"performance": {"phase1_time_s": 100}})
        runs = read_runs(log_path)
        assert runs[0].performance.start_time == "t0"
        assert runs[0].performance.phase1_time_s == 100

    def test_adds_new_field(self, log_path):
        write_runs(log_path, [RunEntry(run=1, status="running")])
        update_run_entry(log_path, 1, {"zarr_config": {"format": "v3"}})
        runs = read_runs(log_path)
        assert runs[0].zarr_config.format == "v3"

    def test_replaces_non_dict_with_dict(self, log_path):
        write_runs(log_path, [RunEntry(run=1, notes="")])
        update_run_entry(log_path, 1, {"notes": "some error"})
        runs = read_runs(log_path)
        assert runs[0].notes == "some error"

    def test_targets_correct_run(self, log_path):
        write_runs(log_path, [
            RunEntry(run=1, status="completed"),
            RunEntry(run=2, status="running"),
        ])
        update_run_entry(log_path, 2, {"status": "phase2"})
        runs = read_runs(log_path)
        assert runs[0].status == "completed"
        assert runs[1].status == "phase2"

    def test_noop_for_missing_run(self, log_path):
        write_runs(log_path, [RunEntry(run=1, status="running")])
        update_run_entry(log_path, 99, {"status": "failed"})
        runs = read_runs(log_path)
        assert runs == [RunEntry(run=1, status="running")]


class TestProgressiveRunLog:
    """Simulate the full progressive write lifecycle."""

    def test_full_lifecycle(self, log_path):
        # Step 1: Initial entry before h5ad read
        runs = read_runs(log_path)
        run_number = get_next_run_number(runs)
        assert run_number == 1
        runs.append(RunEntry(
            run=run_number,
            status="running",
            script_args={"input": "data.h5ad"},
            dataset=RunDataset(input_size_gb=30.4),
            performance=RunPerformance(start_time="2026-03-01T00:00:00+00:00"),
            notes="",
        ))
        write_runs(log_path, runs)

        saved = read_runs(log_path)
        assert len(saved) == 1
        assert saved[0].status == "running"
        assert saved[0].zarr_config is None

        # Step 2: Update after h5ad read
        update_run_entry(log_path, 1, {
            "zarr_config": {"format": "v3", "chunk_shape": [4264929, 1]},
            "dataset": {"n_obs": 4264929, "n_vars": 28476, "input_size_gb": 30.4},
        })

        saved = read_runs(log_path)
        assert saved[0].zarr_config.chunk_shape == [4264929, 1]
        assert saved[0].dataset.n_obs == 4264929

        # Step 3: Phase 1 complete
        update_run_entry(log_path, 1, {
            "status": "phase2",
            "performance": {"phase1_time_s": 507},
        })

        saved = read_runs(log_path)
        assert saved[0].status == "phase2"
        assert saved[0].performance.start_time == "2026-03-01T00:00:00+00:00"
        assert saved[0].performance.phase1_time_s == 507

        # Step 4: Completed
        update_run_entry(log_path, 1, {
            "status": "completed",
            "performance": {
                "end_time": "2026-03-01T01:00:00+00:00",
                "phase2_time_s": 3001,
                "total_time_s": 3508,
            },
        })

        saved = read_runs(log_path)
        assert saved[0].status == "completed"
        assert saved[0].performance.phase1_time_s == 507
        assert saved[0].performance.phase2_time_s == 3001
        assert saved[0].performance.total_time_s == 3508

    def test_failure_lifecycle(self, log_path):
        runs = [RunEntry(run=1, status="running", performance=RunPerformance(start_time="t0"), notes="")]
        write_runs(log_path, runs)

        update_run_entry(log_path, 1, {
            "status": "failed",
            "performance": {"end_time": "t1"},
            "notes": "Error: out of memory",
        })

        saved = read_runs(log_path)
        assert saved[0].status == "failed"
        assert saved[0].notes == "Error: out of memory"
        assert saved[0].performance.start_time == "t0"
        assert saved[0].performance.end_time == "t1"
