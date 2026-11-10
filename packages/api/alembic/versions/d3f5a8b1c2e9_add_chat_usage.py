"""add chat_usage

Revision ID: d3f5a8b1c2e9
Revises: c7a1e9d2f4b8
Create Date: 2026-11-10 23:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'd3f5a8b1c2e9'
down_revision: Union[str, Sequence[str], None] = 'c7a1e9d2f4b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'chat_usage',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('kind', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('key', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('window_start', sa.DateTime(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('kind', 'key', 'window_start', name='uq_chat_usage_kind_key_window'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('chat_usage')
