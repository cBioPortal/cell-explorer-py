"""Pydantic mirror of cbioportal-cell-explorer/packages/highperformer/src/config/schema.ts.

Hand-maintained. The drift test in tests/test_schema_app_config.py reads
the TS source and asserts the field sets match.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class FilterModel(BaseModel):
    obsColumn: str
    ids: list[str]


class AppConfigModel(BaseModel):
    """Full AppConfig. Every field optional so partials validate cleanly."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    dataset: str | None = None
    embedding: str | None = None
    colorBy: Literal["gene", "category"] | None = None
    gene: str | None = None
    category: str | None = None
    geneLabelColumn: str | None = None
    filter: FilterModel | None = None
    summaryObsColumns: list[str] | None = None
    summaryGenes: list[str] | None = None
    showHeader: bool | None = None
    showLeftSidebar: bool | None = None
    showRightSidebar: bool | None = None
    showDatasetDropdown: bool | None = None


def validate_partial(payload: dict) -> AppConfigModel:
    """Validate a partial AppConfig. Raises pydantic.ValidationError on failure."""
    return AppConfigModel.model_validate(payload)


__all__ = ["AppConfigModel", "FilterModel", "validate_partial", "ValidationError"]
