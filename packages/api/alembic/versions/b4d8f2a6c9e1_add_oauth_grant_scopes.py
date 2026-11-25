"""add oauth grant scopes

Revision ID: b4d8f2a6c9e1
Revises: a7c3e9d1b5f2
Create Date: 2026-11-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4d8f2a6c9e1'
down_revision: Union[str, Sequence[str], None] = 'a7c3e9d1b5f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('oauth_grants', sa.Column('scopes', sa.JSON(), nullable=True, server_default=sa.text("'[]'")))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('oauth_grants', 'scopes')
