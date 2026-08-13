"""SQLModel table definitions for the dataset catalog."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, JSON, Relationship, SQLModel, Column


class DatasourceType(str, enum.Enum):
    """Supported datasource types."""

    S3_CLOUDFRONT = "s3_cloudfront"
    HTTP_TOKEN = "http_token"


class Datasource(SQLModel, table=True):
    """A storage backend where zarr datasets are hosted."""

    __tablename__ = "datasources"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    type: DatasourceType
    base_url: str
    # Optional URL used for *server-side* fetches (chat agent, etc.) when the
    # deployment topology means the API can't reach `base_url` directly (e.g.
    # docker-compose where the api container can't hit host-port-mapped
    # localhost:8002 but can reach zarr-server:8000 on the compose network).
    # Falls back to base_url when None.
    internal_base_url: str | None = None
    credential_ref: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    datasets: list["Dataset"] = Relationship(back_populates="datasource")

    @property
    def fetch_base_url(self) -> str:
        """URL the API itself should use to fetch from this datasource."""
        return self.internal_base_url or self.base_url


class Collection(SQLModel, table=True):
    """A study or grouping that datasets belong to."""

    __tablename__ = "collections"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    slug: str = Field(unique=True, index=True)
    description: str | None = None
    publication_url: str | None = None
    publication_citation: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    datasets: list["Dataset"] = Relationship(back_populates="collection")


class Dataset(SQLModel, table=True):
    """A single zarr dataset within a datasource."""

    __tablename__ = "datasets"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    datasource_id: uuid.UUID = Field(foreign_key="datasources.id")
    # Nullable: datasets predating collections stay valid, and deleting a
    # collection orphans its datasets rather than deleting them.
    collection_id: uuid.UUID | None = Field(
        default=None, foreign_key="collections.id", index=True, ondelete="SET NULL"
    )
    name: str
    slug: str = Field(unique=True, index=True)
    path: str
    description: str | None = None
    is_public: bool = Field(default=False)
    required_roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    prompt_addendum: str | None = Field(default=None)
    default_view: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    chat_enabled: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    datasource: Datasource | None = Relationship(back_populates="datasets")
    collection: Collection | None = Relationship(back_populates="datasets")

    @property
    def url(self) -> str | None:
        """Construct the full dataset URL from datasource base_url + path."""
        if self.datasource is None:
            return None
        return f"{self.datasource.base_url}/{self.path}"


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-naive datetime.

    SQLite stores datetimes without timezone info; using naive UTC
    keeps in-memory values consistent with what the DB returns.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ChatThread(SQLModel, table=True):
    """A single chat conversation owned by one user, scoped to one dataset."""

    __tablename__ = "chat_threads"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_sub: str = Field(index=True)  # Keycloak `sub` claim
    dataset_id: uuid.UUID = Field(foreign_key="datasets.id", index=True)
    title: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ChatMessageRow(SQLModel, table=True):
    """One persisted chat message belonging to a thread."""

    __tablename__ = "chat_messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: uuid.UUID = Field(foreign_key="chat_threads.id", index=True, ondelete="CASCADE")
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=_utcnow)
    # Set on assistant rows when Langfuse tracing is enabled. Used by the
    # feedback PUT route to forward thumbs ratings to Langfuse Scores.
    langfuse_trace_id: str | None = Field(default=None, index=True)


class ChatFeedback(SQLModel, table=True):
    """Per-user thumbs feedback on one assistant chat message."""

    __tablename__ = "chat_feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_sub", name="uq_chat_feedback_message_user"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    message_id: uuid.UUID = Field(
        foreign_key="chat_messages.id", index=True, ondelete="CASCADE"
    )
    user_sub: str = Field(index=True)
    rating: str  # "up" | "down"; validated by Pydantic at the route layer
    comment: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
