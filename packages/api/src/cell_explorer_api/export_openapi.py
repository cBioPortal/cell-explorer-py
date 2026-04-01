"""Export the OpenAPI spec as JSON.

Usage:
    uv run python -m cell_explorer_api.export_openapi > openapi.json
"""

import json
import sys

from cell_explorer_api.main import app


def main():
    schema = app.openapi()
    json.dump(schema, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
