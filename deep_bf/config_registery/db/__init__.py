from .connection import create_connection, default_db_path, ensure_schema_initialized


def apply_migrations(*args, **kwargs):
    from .migrate import apply_migrations as _apply_migrations

    return _apply_migrations(*args, **kwargs)


__all__ = [
    "apply_migrations",
    "create_connection",
    "default_db_path",
    "ensure_schema_initialized",
]
