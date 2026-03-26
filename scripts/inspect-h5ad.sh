#!/bin/bash
# Inspect top-level keys and their contents in an h5ad file.
# Usage: bash scripts/inspect-h5ad.sh path/to/file.h5ad

if [ -z "$1" ]; then
    echo "Usage: bash scripts/inspect-h5ad.sh <file.h5ad>"
    exit 1
fi

uv run python -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    for key in f.keys():
        if hasattr(f[key], 'keys'):
            print(f'{key}: {list(f[key].keys())}')
        else:
            print(f'{key}: {f[key].shape} {f[key].dtype}')
" "$1"
