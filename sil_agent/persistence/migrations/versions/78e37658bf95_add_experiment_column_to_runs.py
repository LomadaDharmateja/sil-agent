"""add experiment column to runs

Phase 2 groups runs into experiments — one ablation is 45 runs that belong
together — so the report can ask for "every run in `phase2-main`" instead of
inferring membership from timestamps.

This is the first schema change against tables that already hold rows, which is
what the migration machinery set up in Phase 1 was for. Two properties make it a
safe one: the column is nullable, so existing rows need no value invented for
them, and adding a nullable column with no default is a metadata-only operation
in Postgres — it does not rewrite the table, so it stays fast however many rows
are already there.

Revision ID: 78e37658bf95
Revises: f2f7ecff19b6
Create Date: 2026-08-05 17:23:05.857013

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "78e37658bf95"
down_revision: str | Sequence[str] | None = "f2f7ecff19b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("runs", sa.Column("experiment", sa.String(length=64), nullable=True))
    # Indexed because every report query filters on it, and because the runs
    # table accumulates across every experiment ever executed.
    op.create_index(op.f("ix_runs_experiment"), "runs", ["experiment"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_runs_experiment"), table_name="runs")
    op.drop_column("runs", "experiment")
