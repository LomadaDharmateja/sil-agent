"""Shared test fixtures.

The database-backed tests need a real Postgres, because the two behaviours they
exist to prove — JSONB round-tripping and ``ON CONFLICT DO NOTHING`` on a
composite key — are Postgres behaviours. Testing them against SQLite would prove
that SQLite works.

They run against ``TEST_DATABASE_URL``, a different database from the one the
CLI uses, because these fixtures truncate tables between tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from sil_agent.agent.state import BudgetState, Goal, RunState, RunStatus, utcnow
from sil_agent.persistence.db import make_session_factory
from sil_agent.persistence.repo import RunRepository
from sil_agent.simulators.toy import ToySimulator


@pytest.fixture(scope="session")
def test_database_url() -> str:
    load_dotenv()
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; skipping database tests")
    return url


@pytest.fixture(scope="session")
def session_factory(test_database_url: str) -> sessionmaker[Session]:
    factory = make_session_factory(test_database_url)
    try:
        with factory() as session:
            # Confirms both that Postgres is up and that migrations have run.
            session.execute(text("SELECT 1 FROM episodes LIMIT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(
            "test database not reachable or not migrated "
            f"(run `docker compose up -d` and `alembic upgrade head`): {exc}"
        )
    return factory


def _truncate(factory: sessionmaker[Session]) -> None:
    with factory() as session:
        session.execute(text("TRUNCATE TABLE episodes, runs CASCADE"))
        session.commit()


@pytest.fixture
def repo(session_factory: sessionmaker[Session]) -> Iterator[RunRepository]:
    """A repository pointed at the test database, with clean tables either side.

    This is dependency injection paying off: the repository takes its session
    factory as an argument, so aiming it at a different database is a different
    constructor call rather than a configuration switch.
    """
    _truncate(session_factory)
    yield RunRepository(session_factory)
    _truncate(session_factory)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_run_state(
    *,
    goal: Goal | None = None,
    episodes: int = 20,
    seed: int = 42,
    run_id: UUID | None = None,
    created_at: datetime | None = None,
) -> RunState:
    """A fresh RunState, with sensible defaults for tests."""
    now = created_at or utcnow()
    return RunState(
        run_id=run_id or uuid4(),
        goal=goal or ToySimulator.from_name("branin").default_goal(),
        status=RunStatus.PENDING,
        history=[],
        best=None,
        budget=BudgetState(max_evaluations=episodes),
        step_idx=0,
        seed=seed,
        created_at=now,
        updated_at=now,
    )
