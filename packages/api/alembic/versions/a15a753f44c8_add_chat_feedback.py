"""add chat feedback

Revision ID: a15a753f44c8
Revises: db60f1496e9a
Create Date: 2026-05-14 08:40:04.055833

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'a15a753f44c8'
down_revision: Union[str, Sequence[str], None] = 'db60f1496e9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'chat_feedback',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('message_id', sa.Uuid(), nullable=False),
        sa.Column('user_sub', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('rating', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('comment', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['chat_messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('message_id', 'user_sub', name='uq_chat_feedback_message_user'),
    )
    op.create_index(op.f('ix_chat_feedback_message_id'), 'chat_feedback', ['message_id'], unique=False)
    op.create_index(op.f('ix_chat_feedback_user_sub'), 'chat_feedback', ['user_sub'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_chat_feedback_user_sub'), table_name='chat_feedback')
    op.drop_index(op.f('ix_chat_feedback_message_id'), table_name='chat_feedback')
    op.drop_table('chat_feedback')
