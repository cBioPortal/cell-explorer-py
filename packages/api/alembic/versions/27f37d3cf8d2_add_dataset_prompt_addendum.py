"""add dataset prompt_addendum

Revision ID: 27f37d3cf8d2
Revises: 092e2aa153ce
Create Date: 2026-05-21 10:09:30.488887

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '27f37d3cf8d2'
down_revision: Union[str, Sequence[str], None] = '092e2aa153ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('datasets', sa.Column('prompt_addendum', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('datasets', 'prompt_addendum')
