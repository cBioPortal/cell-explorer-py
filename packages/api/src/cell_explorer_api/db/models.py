"""SQLModel table definitions for the dataset catalog."""

import enum
import uuid
from datetime import datetime, timezone

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


class Dataset(SQLModel, table=True):
    """A single zarr dataset within a datasource."""

    __tablename__ = "datasets"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    datasource_id: uuid.UUID = Field(foreign_key="datasources.id")
    name: str
    slug: str = Field(unique=True, index=True)
    path: str
    description: str | None = None
    is_public: bool = Field(default=False)
    required_roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    chat_enabled: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    datasource: Datasource | None = Relationship(back_populates="datasets")

    @property
    def url(self) -> str | None:
        """Construct the full dataset URL from datasource base_url + path."""
        if self.datasource is None:
            return None
        return f"{self.datasource.base_url}/{self.path}"
