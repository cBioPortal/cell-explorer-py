"""Tables for the OAuth server that signs MCP clients in.

Tokens, authorization codes and MCP keys are stored only as SHA-256 hex
digests. No identity-provider (Keycloak/Entra) tokens are stored. The one
plaintext secret is a dynamically registered confidential client's
``client_secret`` in ``oauth_clients.info``, which the MCP SDK compares as-is.
"""

import uuid
from datetime import datetime

from sqlmodel import JSON, Column, Field, SQLModel

from cell_explorer_api.db.models import utcnow


class OAuthClient(SQLModel, table=True):
    """A dynamically registered OAuth client."""

    __tablename__ = "oauth_clients"

    client_id: str = Field(primary_key=True)
    # OAuthClientInformationFull.model_dump(mode="json"); a confidential client's
    # client_secret is kept here in plaintext, as the SDK's client auth requires.
    info: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class McpResource(SQLModel, table=True):
    """An MCP server (resource) allowed to use this OAuth server."""

    __tablename__ = "mcp_resources"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    url: str = Field(unique=True)
    name: str
    key_hash: str = Field(unique=True)
    created_at: datetime = Field(default_factory=utcnow)
    revoked_at: datetime | None = None


class OAuthPendingAuthorization(SQLModel, table=True):
    """An /authorize request awaiting the identity-provider sign-in."""

    __tablename__ = "oauth_pending_authorizations"

    id: str = Field(primary_key=True)
    client_id: str
    # AuthorizationParams.model_dump(mode="json")
    params: dict = Field(default_factory=dict, sa_column=Column(JSON))
    csrf: str
    expires_at: datetime


class OAuthGrant(SQLModel, table=True):
    """A user's consent for one client on one resource; tokens hang off it."""

    __tablename__ = "oauth_grants"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_sub: str = Field(index=True)
    user_email: str
    user_name: str
    roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    scopes: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # requested at /authorize
    client_id: str
    resource: str
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    revoked_at: datetime | None = None


class OAuthCode(SQLModel, table=True):
    """A single-use authorization code (hash only)."""

    __tablename__ = "oauth_codes"

    code_hash: str = Field(primary_key=True)
    grant_id: uuid.UUID = Field(foreign_key="oauth_grants.id", ondelete="CASCADE")
    client_id: str
    code_challenge: str
    redirect_uri: str
    redirect_uri_provided_explicitly: bool
    resource: str
    expires_at: datetime


class OAuthToken(SQLModel, table=True):
    """An access, refresh or API token (hash only)."""

    __tablename__ = "oauth_tokens"

    token_hash: str = Field(primary_key=True)
    kind: str  # "access" | "refresh" | "api"
    grant_id: uuid.UUID = Field(foreign_key="oauth_grants.id", index=True, ondelete="CASCADE")
    client_id: str
    audience: str
    expires_at: datetime
    used_at: datetime | None = None  # refresh-token rotation
    created_at: datetime = Field(default_factory=utcnow)
