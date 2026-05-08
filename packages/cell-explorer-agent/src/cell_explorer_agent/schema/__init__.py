"""Public API for the AppConfig schema.

The underlying Pydantic models in `app_config.py` are generated from
`schema/app_config.schema.json` via `make generate-app-config`. Do NOT
edit `app_config.py` by hand — your changes will be overwritten.

This module re-exports the generated classes under stable names so call
sites in `tools/ui_action/*` don't need to change when codegen tweaks
internal class names.
"""

from pydantic import ValidationError

from cell_explorer_agent.schema.app_config import (
    Filter as FilterModel,
    Model as AppConfigModel,
)


def validate_partial(payload: dict) -> AppConfigModel:
    """Validate a partial AppConfig. Raises pydantic.ValidationError on failure."""
    return AppConfigModel.model_validate(payload)


__all__ = ["AppConfigModel", "FilterModel", "validate_partial", "ValidationError"]
