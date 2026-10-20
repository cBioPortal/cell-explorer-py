"""Copy a SQLite catalog into an empty Postgres database migrated to the same revision.

One-off move of a deployment's catalog off SQLite. Every table is copied in
foreign-key order with its ids unchanged (all primary keys are UUIDs, so no
sequences need resetting), then read back and compared row by row. The copy
is one transaction: on any failure nothing is written, and it can be re-run.

Usage:
    cd packages/api
    DATABASE_URL=postgresql+asyncpg://... uv run alembic upgrade head
    DATABASE_URL=postgresql+asyncpg://... uv run python -m cell_explorer_api.db.copy_to_postgres snapshot.db

The target URL comes from DATABASE_URL, the same variable the app and Alembic
read, so the password never appears in the command line.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import Table, func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlmodel import SQLModel

from cell_explorer_api.db import create_engine
from cell_explorer_api.db import models  # noqa: F401 — register tables

BATCH_SIZE = 500


class CopyError(Exception):
    """The source or target is not in a state the copy can start from."""


def _tables() -> list[Table]:
    return list(SQLModel.metadata.sorted_tables)


async def _revision(conn: AsyncConnection) -> str | None:
    result = await conn.execute(text("SELECT version_num FROM alembic_version"))
    return result.scalar_one_or_none()


async def _count(conn: AsyncConnection, table: Table) -> int:
    return (await conn.execute(select(func.count()).select_from(table))).scalar_one()


async def _rows(conn: AsyncConnection, table: Table) -> list[dict]:
    """All rows, sorted in Python so both databases yield the same order."""
    pk = [c.name for c in table.primary_key.columns]
    rows = [dict(row._mapping) for row in await conn.execute(select(table))]
    return sorted(rows, key=lambda r: tuple(str(r[k]) for k in pk))


async def _orphans(conn: AsyncConnection) -> list[str]:
    """Rows whose foreign key points at a missing parent.

    SQLite only enforces foreign keys when the connection enables them, so an
    old database can hold rows Postgres would reject.
    """
    found = []
    for table in _tables():
        for fk in table.foreign_keys:
            child, parent = fk.parent, fk.column
            missing = (
                select(func.count())
                .select_from(table)
                .where(child.is_not(None))
                .where(child.not_in(select(parent)))
            )
            n = (await conn.execute(missing)).scalar_one()
            if n:
                found.append(f"{table.name}.{child.name}: {n} row(s) → missing {parent.table.name}")
    return found


async def copy(sqlite_url: str, postgres_url: str) -> dict[str, int]:
    """Copy every table; return the row count per table."""
    source_engine = create_engine(sqlite_url)
    target_engine = create_engine(postgres_url)
    try:
        async with source_engine.connect() as source, target_engine.begin() as target:
            source_rev, target_rev = await _revision(source), await _revision(target)
            if source_rev is None or source_rev != target_rev:
                raise CopyError(
                    f"revision mismatch: source {source_rev}, target {target_rev}; "
                    "run `alembic upgrade head` against both first"
                )
            not_empty = [t.name for t in _tables() if await _count(target, t)]
            if not_empty:
                raise CopyError(f"target tables are not empty: {', '.join(not_empty)}")
            orphans = await _orphans(source)
            if orphans:
                raise CopyError("source has orphaned rows:\n  " + "\n  ".join(orphans))

            counts = {}
            for table in _tables():
                rows = await _rows(source, table)
                for start in range(0, len(rows), BATCH_SIZE):
                    await target.execute(insert(table), rows[start : start + BATCH_SIZE])
                if await _rows(target, table) != rows:
                    raise CopyError(f"{table.name}: rows read back from Postgres differ")
                counts[table.name] = len(rows)
            return counts
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite_path", type=Path, help="SQLite catalog file (a snapshot, not the live file)")
    args = parser.parse_args()

    postgres_url = os.environ.get("DATABASE_URL", "")
    if not postgres_url.startswith("postgresql"):
        sys.exit("DATABASE_URL must be the target postgresql+asyncpg:// URL")
    if not args.sqlite_path.is_file():
        sys.exit(f"no such file: {args.sqlite_path}")

    try:
        counts = asyncio.run(copy(f"sqlite+aiosqlite:///{args.sqlite_path}", postgres_url))
    except CopyError as e:
        sys.exit(f"copy aborted, nothing written: {e}")
    for name, n in counts.items():
        print(f"{name:20} {n:>8}")
    print("copied and verified")


if __name__ == "__main__":
    main()
