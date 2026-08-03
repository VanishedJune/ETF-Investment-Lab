"""SQLAlchemy engine and session factory helpers for the local SQLite file."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..core.paths import database_path as default_database_path
from .fixed_point import register_fixed_point_sql_functions


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
    """Enable SQLite foreign-key constraints for every database connection."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
        register_fixed_point_sql_functions(dbapi_connection)


def sqlite_database_url(path: Path | str | None = None) -> str:
    """Build the SQLite URL for a project-local or explicitly supplied path."""
    target = Path(path) if path is not None else default_database_path()
    return f"sqlite+pysqlite:///{target.resolve().as_posix()}"


def create_database_engine(path: Path | str | None = None) -> Engine:
    """Create an engine for the local SQLite database, making its parent first."""
    target = Path(path) if path is not None else default_database_path()
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        sqlite_database_url(target),
        connect_args={"check_same_thread": False, "timeout": 30},
    )


def create_session_factory(path: Path | str | None = None) -> sessionmaker[Session]:
    """Return a session factory bound to the requested SQLite file."""
    return sessionmaker(bind=create_database_engine(path), expire_on_commit=False)
