"""Tests for StrataConfig + YAML loading."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from cell2zarr.strata.config import StrataConfig


class TestStrataConfigShape:
    def test_minimum_valid_config(self):
        config = StrataConfig(atomic_axes=["cell_type"])
        assert config.atomic_axes == ["cell_type"]
        assert config.coarse == []
        assert config.max_atomic_cardinality == 50_000
        assert config.force is False

    def test_full_config(self):
        config = StrataConfig(
            atomic_axes=["cell_type", "donor", "condition", "treatment"],
            coarse=[["cell_type"], ["cell_type", "treatment"]],
            max_atomic_cardinality=10_000,
            force=True,
        )
        assert config.coarse == [["cell_type"], ["cell_type", "treatment"]]
        assert config.max_atomic_cardinality == 10_000
        assert config.force is True

    def test_atomic_axes_must_be_nonempty(self):
        with pytest.raises(ValidationError):
            StrataConfig(atomic_axes=[])

    def test_max_atomic_cardinality_must_be_positive(self):
        with pytest.raises(ValidationError):
            StrataConfig(atomic_axes=["cell_type"], max_atomic_cardinality=0)


class TestCoarseSubsetValidator:
    def test_coarse_subset_of_atomic_accepted(self):
        StrataConfig(
            atomic_axes=["cell_type", "donor"],
            coarse=[["cell_type"]],
        )

    def test_coarse_axes_outside_atomic_rejected(self):
        with pytest.raises(ValidationError, match="treatment"):
            StrataConfig(
                atomic_axes=["cell_type", "donor"],
                coarse=[["cell_type", "treatment"]],
            )


class TestYamlLoader:
    @pytest.fixture
    def yaml_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "strata.yaml"
        path.write_text(
            "atomic_axes:\n"
            "  - cell_type\n"
            "  - donor\n"
            "coarse:\n"
            "  - cell_type\n"           # bare string — normalize to single-element list
            "  - [cell_type, donor]\n"  # multi-axis as YAML flow list
            "max_atomic_cardinality: 5000\n"
        )
        return path

    def test_from_yaml_loads_fields(self, yaml_path: Path):
        config = StrataConfig.from_yaml(yaml_path)
        assert config.atomic_axes == ["cell_type", "donor"]
        assert config.coarse == [["cell_type"], ["cell_type", "donor"]]
        assert config.max_atomic_cardinality == 5000

    def test_from_yaml_normalizes_single_string_coarse(self, tmp_path: Path):
        path = tmp_path / "single.yaml"
        path.write_text("atomic_axes: [cell_type]\ncoarse:\n  - cell_type\n")
        config = StrataConfig.from_yaml(path)
        assert config.coarse == [["cell_type"]]

    def test_from_yaml_missing_coarse_defaults_empty(self, tmp_path: Path):
        path = tmp_path / "no-coarse.yaml"
        path.write_text("atomic_axes: [cell_type]\n")
        config = StrataConfig.from_yaml(path)
        assert config.coarse == []

    def test_from_yaml_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValidationError):
            StrataConfig.from_yaml(path)
