"""add dataset mcp_enabled

Revision ID: e8b2f4a6c1d3
Revises: d3f5a8b1c2e9
Create Date: 2026-11-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8b2f4a6c1d3'
down_revision: Union[str, Sequence[str], None] = 'd3f5a8b1c2e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('datasets', sa.Column('mcp_enabled', sa.Boolean(), nullable=False, server_default=sa.text('false')))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('datasets', 'mcp_enabled')
