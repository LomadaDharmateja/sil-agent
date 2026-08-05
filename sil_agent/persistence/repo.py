"""The repository — the only code that knows SQL exists.

**Repository pattern**: everything above this file talks in domain objects
(``RunState``, ``Episode``) and never sees a table, a session or a query. That
buys two concrete things here. The loop can be tested against a fake repository
with no database at all, and swapping Postgres for something else later touches
one file.

**Dependency injection**: the session factory is handed to the constructor
rather than imported from a global. "Injection" just means *passed in from
outside instead of reached for from inside*. The payoff is visible in
``tests/conftest.py``: the tests point a repository at ``sil_agent_test`` by
constructing it differently, with no configuration switch, no environment
juggling, and no risk of a test run wiping the development database.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from sil_agent.agent.state import (
    Best,
    BudgetState,
    Episode,
    Frozen,
    Goal,
    RunState,
    RunStatus,
    utcnow,
)
from sil_agent.persistence.models import EpisodeRow, RunRow


class StoredRun(Frozen):
    """A run as it came out of the database.

    ``simulator`` and ``strategy`` travel alongside the state because resuming
    needs to rebuild both, and neither belongs inside the domain ``Goal``.
    ``experiment`` is how the Phase 2 report finds the runs belonging to one
    ablation.
    """

    state: RunState
    simulator: str
    strategy: str
    experiment: str | None = None


class RunRepository:
    """Reads and writes runs and episodes."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    # -- writes ------------------------------------------------------------

    def save_run(
        self,
        state: RunState,
        *,
        simulator: str,
        strategy: str,
        experiment: str | None = None,
    ) -> None:
        """Insert or update the run snapshot.

        An *upsert*: one statement that inserts if the row is new and updates if
        it already exists. Doing it in a single statement rather than
        "SELECT, then INSERT or UPDATE" means there is no window between the two
        in which another process could act, and it is one round trip instead of
        two.

        ``experiment`` is deliberately absent from the update clause below: it
        is set when the run is created and never changed afterwards. That
        matters because ``run_loop`` calls this after every episode without
        knowing anything about experiments, and would otherwise overwrite the
        label with ``None`` on the first save.
        """
        values: dict[str, Any] = {
            "run_id": state.run_id,
            "status": state.status.value,
            "goal": state.goal.model_dump(mode="json"),
            "best": state.best.model_dump(mode="json") if state.best else None,
            "budget": state.budget.model_dump(mode="json"),
            "step_idx": state.step_idx,
            "seed": state.seed,
            "simulator": simulator,
            "strategy": strategy,
            "experiment": experiment,
            "created_at": state.created_at,
            "updated_at": utcnow(),
        }

        statement = insert(RunRow).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["run_id"],
            set_={
                "status": statement.excluded.status,
                "best": statement.excluded.best,
                "budget": statement.excluded.budget,
                "step_idx": statement.excluded.step_idx,
                "updated_at": statement.excluded.updated_at,
            },
        )

        with self._session_factory() as session:
            session.execute(statement)
            session.commit()

    def append_episode(self, run_id: UUID, episode: Episode) -> bool:
        """Append one episode. Returns True if a row was written, False if it already existed.

        **Idempotency** — doing the same thing twice has the same effect as
        doing it once. That property is what makes crash recovery safe: after a
        crash you cannot always tell whether the last write landed, so the only
        safe thing to do is repeat it and require that repeating is harmless.

        It works here because ``(run_id, idx)`` is a *natural key* — it already
        identifies an episode uniquely, so no extra bookkeeping is needed to
        detect the repeat. ``ON CONFLICT DO NOTHING`` turns the second insert
        into a no-op instead of an error, and the caller is told which happened.

        Without this, a worker that died after writing an episode but before
        updating the run snapshot would, on resume, re-run the simulation and
        write a second episode — double-charging the evaluation budget and
        corrupting the very comparison this project exists to make.
        """
        statement = (
            insert(EpisodeRow)
            .values(
                run_id=run_id,
                idx=episode.idx,
                candidate=episode.candidate.model_dump(mode="json"),
                result=episode.result.model_dump(mode="json"),
                evaluation=episode.evaluation.model_dump(mode="json"),
                decision=episode.decision.model_dump(mode="json"),
                cost=episode.cost.model_dump(mode="json"),
                duration_ms=episode.duration_ms,
                created_at=utcnow(),
            )
            .on_conflict_do_nothing(index_elements=["run_id", "idx"])
            # RETURNING, rather than checking rowcount. SQLAlchemy may add its
            # own RETURNING clause to ORM inserts, which makes rowcount
            # unreliable — it reported "already exists" for rows it had just
            # inserted. With an explicit RETURNING the test is unambiguous:
            # ON CONFLICT DO NOTHING returns no row when it skips, one when it
            # inserts.
            .returning(EpisodeRow.idx)
        )

        with self._session_factory() as session:
            written = session.execute(statement).first()
            session.commit()
            return written is not None

    def delete_run(self, run_id: UUID) -> None:
        """Remove a run and its episodes. Used by tests; not exposed on the CLI."""
        with self._session_factory() as session:
            session.execute(delete(EpisodeRow).where(EpisodeRow.run_id == run_id))
            session.execute(delete(RunRow).where(RunRow.run_id == run_id))
            session.commit()

    # -- reads -------------------------------------------------------------

    def load_run(self, run_id: UUID) -> StoredRun | None:
        """Load a run and its full episode history, or None if it does not exist."""
        with self._session_factory() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                return None
            episodes = self._load_episodes(session, run_id)

        state = RunState(
            run_id=row.run_id,
            goal=Goal.model_validate(row.goal),
            status=RunStatus(row.status),
            history=episodes,
            best=Best.model_validate(row.best) if row.best is not None else None,
            budget=BudgetState.model_validate(row.budget),
            step_idx=row.step_idx,
            seed=row.seed,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        return StoredRun(
            state=state,
            simulator=row.simulator,
            strategy=row.strategy,
            experiment=row.experiment,
        )

    def list_runs(self, experiment: str) -> list[StoredRun]:
        """Every run in an experiment, with full episode history, ordered.

        The report is built on this rather than on log files: the episodes table
        is the durable record, so an analysis that reads from it is reproducible
        from the database alone, months later, without the console output that
        produced it still existing.

        Loads eagerly and completely. A 45-run experiment at 200 episodes each
        is 9,000 rows — small enough that streaming or pagination would be
        complexity bought for nothing.
        """
        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(RunRow)
                    .where(RunRow.experiment == experiment)
                    .order_by(RunRow.strategy, RunRow.simulator, RunRow.seed)
                )
                .scalars()
                .all()
            )

            stored: list[StoredRun] = []
            for row in rows:
                episodes = self._load_episodes(session, row.run_id)
                state = RunState(
                    run_id=row.run_id,
                    goal=Goal.model_validate(row.goal),
                    status=RunStatus(row.status),
                    history=episodes,
                    best=Best.model_validate(row.best) if row.best is not None else None,
                    budget=BudgetState.model_validate(row.budget),
                    step_idx=row.step_idx,
                    seed=row.seed,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                stored.append(
                    StoredRun(
                        state=state,
                        simulator=row.simulator,
                        strategy=row.strategy,
                        experiment=row.experiment,
                    )
                )

        return stored

    def load_episodes(self, run_id: UUID) -> list[Episode]:
        """Every episode for a run, ordered by idx. The durable record."""
        with self._session_factory() as session:
            return self._load_episodes(session, run_id)

    def count_episodes(self, run_id: UUID) -> int:
        with self._session_factory() as session:
            total = session.execute(
                select(func.count()).select_from(EpisodeRow).where(EpisodeRow.run_id == run_id)
            ).scalar_one()
            return int(total)

    @staticmethod
    def _load_episodes(session: Session, run_id: UUID) -> list[Episode]:
        rows = (
            session.execute(
                select(EpisodeRow).where(EpisodeRow.run_id == run_id).order_by(EpisodeRow.idx)
            )
            .scalars()
            .all()
        )
        return [
            Episode.model_validate(
                {
                    "idx": row.idx,
                    "candidate": row.candidate,
                    "result": row.result,
                    "evaluation": row.evaluation,
                    "decision": row.decision,
                    "cost": row.cost,
                    "duration_ms": row.duration_ms,
                }
            )
            for row in rows
        ]
