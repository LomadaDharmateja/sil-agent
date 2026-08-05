"""Alembic environment.

**Migrations, briefly**: the database schema is defined by an ordered series of
small scripts, each with an ``upgrade`` and a ``downgrade``. Alembic records
which ones a given database has applied, in a table called ``alembic_version``.
Running ``alembic upgrade head`` brings any database — a colleague's laptop, CI,
production — to the same schema by applying exactly the ones it is missing.

The alternative is creating tables by hand, or calling
``Base.metadata.create_all``. That works right up until the schema needs to
change on a database that already holds rows, at which point there is no record
of what changed or how to apply the same change anywhere else. Hence the project
rule: never create tables by hand.

The database URL comes from the environment, not from ``alembic.ini``, so no
credentials are committed. Set ``ALEMBIC_DATABASE_URL`` to point a migration at
the test database instead of the development one.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from sil_agent.persistence.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata Alembic compares against when autogenerating a migration.
target_metadata = Base.metadata


def _database_url() -> str:
    load_dotenv()
    url = os.environ.get("ALEMBIC_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Neither ALEMBIC_DATABASE_URL nor DATABASE_URL is set. Copy .env.example to .env first."
        )
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it — useful for reviewing a change."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
