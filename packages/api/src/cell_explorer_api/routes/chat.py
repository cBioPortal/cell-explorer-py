"""HTTP routes for the chat agent (Plan 2a)."""

from fastapi import APIRouter

router = APIRouter(tags=["chat"])

# Note: this router is included by routes/__init__.py whose own router has
# prefix="/api". Routes inside use absolute paths ("/chat/{slug}/context"),
# matching the convention in datasets.py.
