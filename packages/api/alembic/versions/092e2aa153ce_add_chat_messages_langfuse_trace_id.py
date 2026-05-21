"""add chat_messages.langfuse_trace_id

Revision ID: 092e2aa153ce
Revises: a15a753f44c8
Create Date: 2026-05-21 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '092e2aa153ce'
down_revision: Union[str, Sequence[str], None] = 'a15a753f44c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('chat_messages') as batch_op:
        batch_op.add_column(
            sa.Column('langfuse_trace_id', sa.String(), nullable=True)
        )
        batch_op.create_index(
            'ix_chat_messages_langfuse_trace_id',
            ['langfuse_trace_id'],
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('chat_messages') as batch_op:
        batch_op.drop_index('ix_chat_messages_langfuse_trace_id')
        batch_op.drop_column('langfuse_trace_id')
