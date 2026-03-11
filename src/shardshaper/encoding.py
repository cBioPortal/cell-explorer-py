"""Encoding config loading and compressor creation."""
import json
from pathlib import Path

from .models import CompressorSpec, EncodingConfig


def resolve_template(value, variables: dict[str, int]):
    """Resolve template strings like '{n_obs}' in encoding config values.

    Returns the original string if the variable is not yet available.
    """
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        key = value[1:-1]
        if key in variables:
            return variables[key]
        return value
    return value


def load_encoding_config(config_path: Path, variables: dict[str, int]) -> EncodingConfig:
    """Load an encoding config JSON file and resolve template variables.

    Template variables like {n_obs}, {n_vars}, {n_dim} are resolved using
    the provided variables dict. Unresolvable templates are kept as strings.
    """
    with open(config_path) as f:
        raw = json.load(f)

    # Resolve templates in the raw dict before pydantic parsing
    resolved = {}
    for section, encoding in raw.items():
        resolved[section] = {}
        for key, val in encoding.items():
            if isinstance(val, list):
                resolved[section][key] = [resolve_template(v, variables) for v in val]
            elif isinstance(val, dict):
                resolved[section][key] = val
            else:
                resolved[section][key] = resolve_template(val, variables)
    return EncodingConfig.model_validate(resolved)


def make_compressor(spec: CompressorSpec | dict | None):
    """Create a zarr codec from a CompressorSpec or dict."""
    if spec is None:
        return "auto"
    import zarr.codecs
    if isinstance(spec, dict):
        spec = CompressorSpec(**spec)
    name = spec.name.lower()
    if name == "zstd":
        return zarr.codecs.ZstdCodec(level=spec.level)
    elif name == "blosc":
        return zarr.codecs.BloscCodec(cname=spec.cname, clevel=spec.level)
    elif name == "gzip":
        return zarr.codecs.GzipCodec(level=spec.level)
    else:
        raise ValueError(f"Unknown compressor: {name}")
