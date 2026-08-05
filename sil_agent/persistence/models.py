"""SQLAlchemy table definitions, per TECHNICAL_DESIGN §4.

Two tables, with different jobs:

``episodes`` is **append-only**. A row is written once and never updated. This
is the durable record — the thing that makes a run reconstructible.

``runs`` is a **rolling snapshot**. It holds the latest status, best and
step_idx so that inspecting a run is one cheap query. It is a convenience, not
the source of truth: the loop recomputes step_idx and best from ``episodes``.

Why JSONB rather than a column per field: the nested shapes (goal, candidate,
result) are defined and validated by Pydantic, and they will grow in later
phases. Storing them as JSONB means adding a field to ``Evaluation`` in Phase 4
needs no migration. JSONB (rather than JSON) is stored parsed and can be indexed
and queried with operators like ``->>``, which Phase 2's analysis will use.

The cost of that choice is real: Postgres will not enforce the shape of these
columns. Pydantic does, on the way in and on the way out.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base — Alembic reads table definitions off this class."""


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    goal: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    best: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    budget: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    step_idx: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Not in the design's table sketch, but required for resume to work.
    #
    # Rule 1 says the next episode is fully determined by persisted state. If the
    # simulator and strategy names live only in the command line that started the
    # run, then `resume --run-id X` cannot reconstruct the run without the user
    # remembering what they typed — and a run resumed against the wrong simulator
    # would silently produce garbage. So they are part of the state.
    simulator: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)

    # Phase 2. Groups the runs belonging to one ablation into a set the report
    # can query: "every run in experiment `phase2-main`". Nullable because runs
    # started from the CLI belong to no experiment, and because a nullable
    # column can be added to a table that already holds rows without needing a
    # value invented for each of them.
    experiment: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EpisodeRow(Base):
    __tablename__ = "episodes"

    # Composite primary key (run_id, idx) — the natural key. See
    # RunRepository.append_episode for why that matters.
    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)

    candidate: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    decision: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    cost: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
