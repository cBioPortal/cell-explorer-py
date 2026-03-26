#!/bin/bash
# Inspect top-level keys and their contents in an h5ad file.
# Usage: bash scripts/inspect-h5ad.sh path/to/file.h5ad

if [ -z "$1" ]; then
    echo "Usage: bash scripts/inspect-h5ad.sh <file.h5ad>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
uv run python "$SCRIPT_DIR/inspect_h5ad.py" "$1"
