import sqlite3
from pathlib import Path

from ..errors import SchemaNotInitializedError

DEFAULT_DB_NAME = "config_registery.db"
REQUIRED_TABLES = {
    "compounding",
    "data_size",
    "data_type",
    "experiments",
    "resize_gt",
    "samples_organization",
    "transform_data",
    "webdataset_beamformer",
    "model",
    "trainloop",
}


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_DB_NAME


def create_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    resolved_db_path = Path(db_path) if db_path is not None else default_db_path()
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(resolved_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_schema_initialized(
    conn: sqlite3.Connection, db_path: str | Path | None = None
) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    existing_tables = {str(row["name"]) for row in rows}

    missing = REQUIRED_TABLES - existing_tables
    if missing:
        path = str(Path(db_path)) if db_path is not None else str(default_db_path())
        missing_tables = ", ".join(sorted(missing))
        raise SchemaNotInitializedError(
            (
                f"Database schema is not initialized at {path}. "
                f"Missing tables: {missing_tables}. "
                "Run: python -m deep_bf.config_registery.db.migrate"
            )
        )
