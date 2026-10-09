"""add collection attribution fields

Revision ID: b4f1c2d3e5a6
Revises: 16c6d7de8c39
Create Date: 2026-10-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'b4f1c2d3e5a6'
down_revision: Union[str, Sequence[str], None] = '16c6d7de8c39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ("license_name", "license_url", "source_url", "attribution_note")


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("collections") as batch_op:
        for name in _COLUMNS:
            batch_op.add_column(sa.Column(name, sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("collections") as batch_op:
        for name in reversed(_COLUMNS):
            batch_op.drop_column(name)
