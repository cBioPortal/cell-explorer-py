"""add dataset metadata

Revision ID: 437779e95bd8
Revises: aae63119b18e
Create Date: 2026-08-26 14:53:40.039159

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '437779e95bd8'
down_revision: Union[str, Sequence[str], None] = 'aae63119b18e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_metadata",
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("n_obs", sa.Integer(), nullable=True),
        sa.Column("n_vars", sa.Integer(), nullable=True),
        sa.Column("zarr_version", sa.Integer(), nullable=True),
        sa.Column("obsm_keys", sa.JSON(), nullable=True),
        sa.Column("obs_columns", sa.JSON(), nullable=True),
        sa.Column("var_columns", sa.JSON(), nullable=True),
        sa.Column("layers", sa.JSON(), nullable=True),
        sa.Column("x_dtype", sa.String(), nullable=True),
        sa.Column("x_encoding", sa.String(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("dataset_id"),
    )


def downgrade() -> None:
    op.drop_table("dataset_metadata")
