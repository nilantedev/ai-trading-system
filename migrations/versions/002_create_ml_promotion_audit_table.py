"""create ml promotion audit table

Revision ID: 002
Revises: 001
Create Date: 2025-09-07
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    if 'ml_promotion_audit' not in tables:
        op.create_table(
            'ml_promotion_audit',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('model_id', sa.String(length=128), nullable=False),
            sa.Column('symbol', sa.String(length=32), nullable=True),
            sa.Column('model_type', sa.String(length=64), nullable=True),
            sa.Column('decision', sa.String(length=32), nullable=False),
            sa.Column('timestamp', sa.DateTime(timezone=False), nullable=False),
            sa.Column('details', sa.JSON(), nullable=True),
        )
    # Create indexes if missing
    existing_indexes = {idx['name'] for idx in inspector.get_indexes('ml_promotion_audit')}
    index_specs = [
        ('ix_ml_promotion_audit_model_id', ['model_id']),
        ('ix_ml_promotion_audit_symbol', ['symbol']),
        ('ix_ml_promotion_audit_model_type', ['model_type']),
        ('ix_ml_promotion_audit_decision', ['decision']),
        ('ix_ml_promotion_audit_timestamp', ['timestamp']),
    ]
    for name, cols in index_specs:
        if name not in existing_indexes:
            op.create_index(name, 'ml_promotion_audit', cols)

def downgrade():
    op.drop_table('ml_promotion_audit')
