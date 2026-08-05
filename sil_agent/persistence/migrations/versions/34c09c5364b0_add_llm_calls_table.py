"""add llm_calls table

Stores every prompt sent and every reply received, keyed by (run_id, call_key)
where call_key hashes the request content.

The composite primary key is doing the same work here as (run_id, idx) does on
`episodes`: it is the natural key, so writing a call twice is a no-op via
`ON CONFLICT DO NOTHING` rather than a duplicate. That is what makes a resumed
run reuse the model's earlier answers instead of paying for new ones.

`ondelete=CASCADE` matches `episodes` — deleting a run removes its whole record,
including what the model was asked and what it said.

Revision ID: 34c09c5364b0
Revises: 78e37658bf95
Create Date: 2026-08-05 18:40:29.582679

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "34c09c5364b0"
down_revision: str | Sequence[str] | None = "78e37658bf95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_calls",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("call_key", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("request", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "call_key"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("llm_calls")
