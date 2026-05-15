"""CLI tests via click's CliRunner."""
from pathlib import Path

from click.testing import CliRunner

from cell2zarr.cli import cli


class TestCliNoAxes:
    def test_missing_axes_lists_categoricals(self, tiny_anndata_zarr: Path):
        runner = CliRunner()
        result = runner.invoke(cli, ["build-strata", str(tiny_anndata_zarr)])
        assert result.exit_code != 0
        assert "--atomic-axes" in result.output
        assert "cell_type" in result.output
        assert "donor" in result.output


class TestCliFlags:
    def test_atomic_axes_via_cli_runs(self, tiny_anndata_zarr: Path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["build-strata", str(tiny_anndata_zarr), "--atomic-axes", "cell_type"],
        )
        assert result.exit_code == 0, result.output

    def test_coarse_axes_via_cli_runs(self, tiny_anndata_zarr: Path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build-strata",
                str(tiny_anndata_zarr),
                "--atomic-axes", "cell_type",
                "--atomic-axes", "donor",
                "--coarse", "cell_type",
                "--coarse", "cell_type,donor",
            ],
        )
        assert result.exit_code == 0, result.output


class TestCliForce:
    def test_second_run_without_force_errors(self, tiny_anndata_zarr: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["build-strata", str(tiny_anndata_zarr), "--atomic-axes", "cell_type"],
        )
        result = runner.invoke(
            cli,
            ["build-strata", str(tiny_anndata_zarr), "--atomic-axes", "cell_type"],
        )
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_force_flag_overrides(self, tiny_anndata_zarr: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["build-strata", str(tiny_anndata_zarr), "--atomic-axes", "cell_type"],
        )
        result = runner.invoke(
            cli,
            [
                "build-strata",
                str(tiny_anndata_zarr),
                "--atomic-axes",
                "donor",
                "--force",
            ],
        )
        assert result.exit_code == 0, result.output


class TestCliConfig:
    def test_config_yaml_takes_effect(self, tmp_path: Path, tiny_anndata_zarr: Path):
        config_yaml = tmp_path / "strata.yaml"
        config_yaml.write_text(
            "atomic_axes:\n"
            "  - cell_type\n"
            "  - donor\n"
            "coarse:\n"
            "  - cell_type\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["build-strata", str(tiny_anndata_zarr), "--config", str(config_yaml)],
        )
        assert result.exit_code == 0, result.output

    def test_cli_flag_overrides_yaml(self, tmp_path: Path, tiny_anndata_zarr: Path):
        """If both CLI and YAML set atomic_axes, CLI wins."""
        config_yaml = tmp_path / "strata.yaml"
        config_yaml.write_text("atomic_axes: [donor]\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build-strata",
                str(tiny_anndata_zarr),
                "--config", str(config_yaml),
                "--atomic-axes", "cell_type",
            ],
        )
        assert result.exit_code == 0, result.output

        from cell2zarr.strata.io import open_dataset
        root = open_dataset(tiny_anndata_zarr)
        assert list(root["uns"]["strata"]["atomic"].attrs["axes"]) == ["cell_type"]
