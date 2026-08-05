"""Engine and session construction.

Two small patterns worth naming, because they come up everywhere from here on:

**Session factory.** A SQLAlchemy ``Session`` is a unit of work — a scratchpad
holding objects and a transaction. You do not share one across a whole program;
you open one per operation and close it. ``sessionmaker`` is a factory that
stamps out sessions bound to the same engine.

**Connection pooling.** The ``Engine`` keeps a pool of open TCP connections to
Postgres and hands them out. Opening a connection costs milliseconds and a
round trip; over a 100-episode run that adds up, and every save would pay it.
The pool means the cost is paid once.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_ENV_VAR = "DATABASE_URL"


def resolve_database_url(env_var: str = DEFAULT_ENV_VAR) -> str:
    """Read a database URL from the environment, loading ``.env`` if present.

    No secrets in code — the URL always comes from the environment. ``.env`` is
    gitignored; ``.env.example`` is committed so a new machine knows what to set.
    """
    load_dotenv()
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} is not set. Copy .env.example to .env, or set it in the environment."
        )
    return url


def create_db_engine(url: str, *, echo: bool = False) -> Engine:
    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,  # cheap liveness check; avoids handing out a dead connection
        future=True,
    )


def make_session_factory(url: str, *, echo: bool = False) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(url, echo=echo), expire_on_commit=False)
