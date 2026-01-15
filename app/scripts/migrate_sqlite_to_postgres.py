from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("psycopg is required to migrate data to Postgres.") from exc

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.storage.database import DatabaseManager

TABLE_ORDER = [
    "sessions",
    "session_pending_details",
    "session_short_term_memory",
    "session_images",
    "messages",
    "personal_facts",
    "answer_history",
    "cv_profile",
    "cv_skill",
    "cv_experience",
    "cv_experience_technology",
    "cv_experience_responsibility",
    "cv_experience_achievement",
    "cv_education",
    "cv_project",
    "cv_project_technology",
    "cv_project_description",
    "cv_preference_location",
    "memory_items",
    "user_profile",
    "user_settings",
    "memory_events",
    "gmail_notifications",
]

SEQUENCE_TABLES = [
    "messages",
    "session_short_term_memory",
    "session_images",
    "personal_facts",
    "answer_history",
    "cv_skill",
    "cv_experience",
    "cv_experience_technology",
    "cv_experience_responsibility",
    "cv_experience_achievement",
    "cv_education",
    "cv_project",
    "cv_project_technology",
    "cv_project_description",
    "cv_preference_location",
    "memory_items",
    "memory_events",
]


def _default_sqlite_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "atlas.db"


def _parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL.")
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("SQLITE_DB_PATH") or str(_default_sqlite_path()),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL") or "",
        help="PostgreSQL connection string.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target tables before inserting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per batch insert.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=10,
        help="Postgres connection timeout in seconds.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Print progress every N inserted rows (0 to disable).",
    )
    return parser.parse_args()


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _pg_count(connection: psycopg.Connection, table: str) -> int:
    row = connection.execute(
        sql.SQL("SELECT COUNT(*) FROM {table}").format(table=sql.Identifier(table))
    ).fetchone()
    if row is None:
        return 0
    return int(row[0]) if not isinstance(row, dict) else int(next(iter(row.values())))


def main() -> int:
    args = _parse_args()
    sqlite_path = Path(args.sqlite_path).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = (Path.cwd() / sqlite_path).resolve()
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite database not found at {sqlite_path}.")
    database_url = args.database_url or get_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required to migrate to Postgres.")

    print("Connecting to Postgres...", flush=True)
    bootstrap = DatabaseManager(database_url=database_url, connect_timeout=args.connect_timeout)
    bootstrap.initialize()
    bootstrap.close()
    print("Schema ensured.", flush=True)

    print(f"Opening SQLite database at {sqlite_path}...", flush=True)
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row
    pg_conn = psycopg.connect(database_url, connect_timeout=args.connect_timeout)

    sqlite_tables = _sqlite_tables(sqlite_conn)
    with pg_conn:
        if args.truncate:
            truncate_sql = sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                sql.SQL(", ").join(sql.Identifier(table) for table in TABLE_ORDER)
            )
            pg_conn.execute(truncate_sql)

        for table in TABLE_ORDER:
            if table not in sqlite_tables:
                print(f"[skip] {table} (not found in SQLite)")
                continue
            if not args.truncate and _pg_count(pg_conn, table) > 0:
                raise SystemExit(
                    f"Target table '{table}' is not empty. Use --truncate to overwrite."
                )
            print(f"[start] {table}", flush=True)
            columns = _sqlite_columns(sqlite_conn, table)
            if not columns:
                print(f"[skip] {table} (no columns)")
                continue
            columns_sql = ", ".join(f'"{col}"' for col in columns)
            select_sql = f"SELECT {columns_sql} FROM {table}"
            insert_sql = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
                table=sql.Identifier(table),
                columns=sql.SQL(", ").join(sql.Identifier(col) for col in columns),
                values=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
            )

            cursor = sqlite_conn.execute(select_sql)
            total = 0
            while True:
                rows = cursor.fetchmany(args.batch_size)
                if not rows:
                    break
                with pg_conn.cursor() as pg_cursor:
                    pg_cursor.executemany(insert_sql, [tuple(row) for row in rows])
                pg_conn.commit()
                total += len(rows)
                if args.log_every and total % args.log_every == 0:
                    print(f"[progress] {table}: {total} rows", flush=True)
            print(f"[ok] {table}: {total} rows")

        for table in SEQUENCE_TABLES:
            pg_conn.execute(
                sql.SQL(
                    """
                    SELECT setval(
                        pg_get_serial_sequence({table_literal}, 'id'),
                        COALESCE(MAX(id), 1),
                        MAX(id) IS NOT NULL
                    )
                    FROM {table_ident}
                    """
                ).format(
                    table_literal=sql.Literal(table),
                    table_ident=sql.Identifier(table),
                )
            )

    sqlite_conn.close()
    pg_conn.close()
    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
