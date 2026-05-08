import argparse
import re
import sqlite3
from pathlib import Path

from .connection import create_connection, default_db_path

MIGRATION_FILE_RE = re.compile(r"^(\d+)_([a-zA-Z0-9_\-]+)\.sql$")

def ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def parse_migration_file(path: Path) -> tuple[int, str] | None:
    match = MIGRATION_FILE_RE.match(path.name)
    if not match:
        return None

    version = int(match.group(1))
    name = match.group(2)
    return version, name


def migration_applied(conn: sqlite3.Connection, version: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?",
        (version,),
    ).fetchone()
    return row is not None


def apply_migrations(
    db_path: str | Path | None = None, migrations_dir: str | Path | None = None
) -> None:
    conn = create_connection(db_path)
    try:
        ensure_migrations_table(conn)

        default_dir = Path(__file__).resolve().parent / "migrations"
        migrations_root = (
            Path(migrations_dir) if migrations_dir is not None else default_dir
        )
        files = sorted(migrations_root.glob("*.sql"))

        for migration_file in files:
            parsed = parse_migration_file(migration_file)
            if parsed is None:
                continue

            version, name = parsed
            if migration_applied(conn, version):
                continue

            sql = migration_file.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, name),
            )
            conn.commit()
            print(f"Applied migration {version:04d}: {name}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", dest="db_path", type=str, default=str(default_db_path())
    )
    parser.add_argument(
        "--migrations-dir", dest="migrations_dir", type=str, default=None
    )
    args = parser.parse_args()

    apply_migrations(db_path=args.db_path, migrations_dir=args.migrations_dir)


if __name__ == "__main__":
    main()
