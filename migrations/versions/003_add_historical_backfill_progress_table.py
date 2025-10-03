"""add historical_backfill_progress table

Revision ID: 003
Revises: 002
Create Date: 2025-09-07
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'historical_backfill_progress',
        sa.Column('symbol', sa.String(length=32), primary_key=True, nullable=False),
        sa.Column('last_date', sa.Date(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('idx_hbp_last_date', 'historical_backfill_progress', ['last_date'])


def downgrade() -> None:
    op.drop_index('idx_hbp_last_date', table_name='historical_backfill_progress')
    op.drop_table('historical_backfill_progress')
