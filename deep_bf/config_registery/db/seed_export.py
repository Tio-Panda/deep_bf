from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .connection import create_connection, default_db_path

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_DELTA_SEEDS_DIR = Path(__file__).resolve().parent / "delta_seeds"
DEFAULT_FULL_SEED_PATH = Path(__file__).resolve().parent / "seeds" / "full_seed.sql"

FULL_EXPORT_TABLE_ORDER = (
    "conv2d_init_config",
    "activation_config",
    "criterion_config",
    "optimizer_config",
    "scheduler_config",
    "hyperparameters_config",
    "data_type_config",
    "beamformer_config",
    "resampler_config",
    "compounding_config",
    "apod_config",
    "data_size_config",
    "data_preprocessing_config",
    "samples_organization_config",
    "beamformer_setup",
    "trainloop_setup",
    "model_pack",
    "architecture_cnn_bf_config",
    "webdataset_beamformer_pack",
    "experiment",
)

TABLE_PK_COLUMNS: dict[str, tuple[str, ...]] = {
    "architecture_cnn_bf_config": ("model_id", "family", "pos"),
}


def _validate_identifier(identifier: str, context: str) -> None:
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(
            f"Invalid {context} '{identifier}'. Only letters, numbers, and underscores are allowed."
        )


def _quote_identifier(identifier: str) -> str:
    _validate_identifier(identifier, "identifier")
    return f'"{identifier}"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(value)

    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"

    raw = str(value).replace("'", "''")
    return f"'{raw}'"


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return [str(row["name"]) for row in rows]


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _resolve_ordered_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    existing = {str(row["name"]) for row in rows}
    existing.discard("schema_migrations")

    ordered = [name for name in FULL_EXPORT_TABLE_ORDER if name in existing]
    ordered_set = set(ordered)
    extras = sorted(existing - ordered_set)
    return [*ordered, *extras]


def _order_by_clause(table_name: str, columns: list[str]) -> str | None:
    if table_name in TABLE_PK_COLUMNS:
        pk_columns = TABLE_PK_COLUMNS[table_name]
        if all(name in columns for name in pk_columns):
            joined = ", ".join(_quote_identifier(name) for name in pk_columns)
            return f" ORDER BY {joined} ASC"

    if "id" in columns:
        return ' ORDER BY "id" ASC'

    return None


def _insert_statement(table_name: str, columns: list[str], row: sqlite3.Row) -> str:
    table_expr = _quote_identifier(table_name)
    columns_expr = ", ".join(_quote_identifier(column) for column in columns)
    values_expr = ", ".join(_sql_literal(row[column]) for column in columns)
    return f"INSERT INTO {table_expr} ({columns_expr}) VALUES ({values_expr});"


def export_full_seed(
    db_path: str | Path | None = None,
    out_file: str | Path | None = None,
) -> Path:
    resolved_out = Path(out_file) if out_file is not None else DEFAULT_FULL_SEED_PATH
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    conn = create_connection(db_path)
    try:
        tables = _resolve_ordered_tables(conn)
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        lines: list[str] = [
            "-- Auto-generated full seed for config_registery.",
            f"-- Generated at {now}",
            "BEGIN;",
            "",
        ]

        for table_name in tables:
            columns = _table_columns(conn, table_name)
            if not columns:
                continue

            table_expr = _quote_identifier(table_name)
            query = f"SELECT * FROM {table_expr}"
            order_by = _order_by_clause(table_name, columns)
            if order_by is not None:
                query += order_by

            rows = conn.execute(query).fetchall()
            if not rows:
                continue

            lines.append(f"-- table: {table_name}")
            for row in rows:
                lines.append(_insert_statement(table_name, columns, row))
            lines.append("")

        lines.append("COMMIT;")
        lines.append("")
        resolved_out.write_text("\n".join(lines), encoding="utf-8")
        return resolved_out
    finally:
        conn.close()


def _normalize_pk(table_name: str, pk_value: int | dict[str, Any]) -> dict[str, Any]:
    if isinstance(pk_value, dict):
        if not pk_value:
            raise ValueError("pk_value dict cannot be empty")
        for key in pk_value:
            _validate_identifier(key, "pk column")
        return pk_value

    pk_columns = TABLE_PK_COLUMNS.get(table_name, ("id",))
    if len(pk_columns) != 1:
        names = ", ".join(pk_columns)
        raise ValueError(
            f"table '{table_name}' requires composite PK {{{names}}}; provide pk_value as dict"
        )

    return {pk_columns[0]: pk_value}


def export_delta_seed(
    table_name: str,
    pk_value: int | dict[str, Any],
    db_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    _validate_identifier(table_name, "table name")
    resolved_out_dir = Path(out_dir) if out_dir is not None else DEFAULT_DELTA_SEEDS_DIR
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    conn = create_connection(db_path)
    try:
        if not _table_exists(conn, table_name):
            raise ValueError(f"table '{table_name}' does not exist")

        columns = _table_columns(conn, table_name)
        if not columns:
            raise ValueError(f"table '{table_name}' does not have columns")

        normalized_pk = _normalize_pk(table_name, pk_value)
        for key in normalized_pk:
            if key not in columns:
                raise ValueError(
                    f"pk column '{key}' does not exist in table '{table_name}'"
                )

        table_expr = _quote_identifier(table_name)
        where_clause = " AND ".join(
            f"{_quote_identifier(name)} = ?" for name in normalized_pk
        )
        row = conn.execute(
            f"SELECT * FROM {table_expr} WHERE {where_clause}",
            tuple(normalized_pk.values()),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"row was not found in table '{table_name}' for pk={normalized_pk}"
            )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        pk_for_name = "_".join(str(normalized_pk[key]) for key in normalized_pk)
        filename = f"{timestamp}_add_{table_name}_{pk_for_name}.sql"
        out_file = resolved_out_dir / filename

        lines = [
            "-- Auto-generated delta seed for config_registery.",
            f"-- table={table_name} pk={normalized_pk}",
            "BEGIN;",
            _insert_statement(table_name, columns, row),
            "COMMIT;",
            "",
        ]
        out_file.write_text("\n".join(lines), encoding="utf-8")
        return out_file
    finally:
        conn.close()


def default_seed_db_path() -> Path:
    return default_db_path()
