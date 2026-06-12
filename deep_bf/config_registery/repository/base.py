import sqlite3
from typing import Any

from ..errors import ConfigNotFoundError, DuplicateConfigError


class BaseRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _insert(self, query: str, params: tuple[Any, ...]) -> int:
        cursor = self.conn.execute(query, params)
        self.conn.commit()
        return int(cursor.lastrowid)

    def _fetch_one(
        self, query: str, params: tuple[Any, ...], context: str
    ) -> sqlite3.Row:
        rows = self.conn.execute(query, params).fetchall()
        if not rows:
            raise ConfigNotFoundError(f"{context} was not found")

        if len(rows) > 1:
            raise DuplicateConfigError(
                f"{context} expected one row and found {len(rows)}"
            )

        return rows[0]
