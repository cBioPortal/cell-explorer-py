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

    `values` and `cardinality` are produced by two different paths with
    different limits, not one shared computation: the catalogue
    (`/api/datasets`) serves values harvested and capped at
    `FACET_VALUE_CAP` (100) at store-discovery time, while chat
    (`/api/chat/{slug}/context`) derives them live from the open store under
    its own separate cap. The two routes can legitimately disagree — e.g. a
    column may report values on one route and not the other, or ontology
    columns may appear on chat's response but never the catalogue's. Do not
    assume the two routes describe the same column identically.
    """

    name: str
    dtype: Literal["categorical", "numeric", "string"]
    cardinality: int | None = None
    values: list[str] | None = None
    facet: str | None = None
