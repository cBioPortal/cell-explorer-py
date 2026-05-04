import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from cell_explorer_agent.schema.app_config import (
    AppConfigModel,
    FilterModel,
    validate_partial,
)

TS_SCHEMA_PATH = (
    Path(__file__).parents[4]
    / "cbioportal-cell-explorer/packages/highperformer/src/config/schema.ts"
)


def test_valid_partial_color_by_gene():
    m = validate_partial({"colorBy": "gene", "gene": "CD8A"})
    assert m.colorBy == "gene"
    assert m.gene == "CD8A"


def test_valid_partial_filter():
    m = validate_partial(
        {"filter": {"obsColumn": "cell_type", "ids": ["c1", "c2"]}}
    )
    assert m.filter is not None
    assert m.filter.obsColumn == "cell_type"


def test_rejects_invalid_colorby_value():
    with pytest.raises(ValidationError):
        validate_partial({"colorBy": "nope"})


def test_filter_model_ids_is_list_of_str():
    f = FilterModel(obsColumn="cell_type", ids=["a", "b"])
    assert f.ids == ["a", "b"]


def test_drift_check_against_ts_schema():
    """Python model field names must match the TS Zod object fields.

    If the TS file is unavailable (e.g. running outside the monorepo),
    the test is skipped rather than failing.
    """
    if not TS_SCHEMA_PATH.is_file():
        pytest.skip(f"TS schema not found at {TS_SCHEMA_PATH}")

    text = TS_SCHEMA_PATH.read_text()

    # Extract the RawConfigSchema z.object({...}) block.
    m = re.search(
        r"RawConfigSchema\s*=\s*z\.object\(\{(.*?)\}\)\.refine",
        text,
        flags=re.DOTALL,
    )
    assert m is not None, "Could not locate RawConfigSchema in TS file"
    block = m.group(1)

    ts_fields = set(re.findall(r"^\s*(\w+)\s*:", block, flags=re.MULTILINE))
    py_fields = set(AppConfigModel.model_fields.keys())

    assert ts_fields == py_fields, (
        f"AppConfigModel fields drift from TS schema.\n"
        f"only in TS: {ts_fields - py_fields}\n"
        f"only in Py: {py_fields - ts_fields}"
    )
