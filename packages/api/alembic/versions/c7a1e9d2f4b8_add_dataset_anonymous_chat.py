"""add dataset anonymous_chat

Revision ID: c7a1e9d2f4b8
Revises: b4f1c2d3e5a6
Create Date: 2026-11-10 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7a1e9d2f4b8'
down_revision: Union[str, Sequence[str], None] = 'b4f1c2d3e5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('datasets', sa.Column('anonymous_chat', sa.Boolean(), nullable=False, server_default=sa.text('false')))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('datasets', 'anonymous_chat')
