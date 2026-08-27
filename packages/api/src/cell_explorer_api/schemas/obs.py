"""Obs column description, shared by the chat context and catalogue responses.

One model, one name: FastAPI derives OpenAPI schema names from the class, and
the frontend generates its client from that document, so a second class with
this name would collide.
"""

from typing import Literal

from pydantic import BaseModel


class ObsColumnInfo(BaseModel):
    """One obs column as described to API callers.

    `name` is always the column's real name in the file. `facet` is the
    canonical facet it was resolved to, or None when it matched no definition —
    the interpretation, kept separate from the fact.
    """

    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None = None
    values: list[str] | None = None
    facet: str | None = None
