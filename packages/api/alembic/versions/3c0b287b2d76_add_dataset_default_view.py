"""add dataset default_view

Revision ID: 3c0b287b2d76
Revises: 27f37d3cf8d2
Create Date: 2026-07-15 14:28:24.803108

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '3c0b287b2d76'
down_revision: Union[str, Sequence[str], None] = '27f37d3cf8d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("default_view", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "default_view")
