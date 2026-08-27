"""add dataset obs facets

Revision ID: 16c6d7de8c39
Revises: 437779e95bd8
Create Date: 2026-08-27 11:56:42.787967

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '16c6d7de8c39'
down_revision: Union[str, Sequence[str], None] = '437779e95bd8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("dataset_metadata", sa.Column("obs_facets", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("dataset_metadata", "obs_facets")
