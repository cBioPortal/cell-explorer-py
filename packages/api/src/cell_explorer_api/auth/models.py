"""Authentication models."""

from pydantic import BaseModel


class User(BaseModel):
    """User identity decoded from a Keycloak JWT."""

    sub: str
    name: str | None = None
    email: str | None = None
    roles: list[str] = []
