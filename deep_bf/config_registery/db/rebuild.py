import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from deep_bf.config_registery.db.connection import (  # type: ignore
        create_connection,
        default_db_path,
    )
    from deep_bf.config_registery.db.migrate import apply_migrations  # type: ignore
    from deep_bf.config_registery.db.seed_export import (  # type: ignore
        DEFAULT_DELTA_SEEDS_DIR,
        DEFAULT_FULL_SEED_PATH,
    )
else:
    from .connection import create_connection, default_db_path  # pragma: no cover
    from .migrate import apply_migrations  # pragma: no cover
    from .seed_export import (  # pragma: no cover
        DEFAULT_DELTA_SEEDS_DIR,
        DEFAULT_FULL_SEED_PATH,
    )


def rebuild_database(
    db_path: str | Path,
    full_seed: str | Path | None = None,
    deltas_dir: str | Path | None = None,
    drop_existing: bool = False,
) -> None:
    resolved_db_path = Path(db_path)
    if drop_existing and resolved_db_path.exists():
        resolved_db_path.unlink()

    apply_migrations(db_path=resolved_db_path)

    conn = create_connection(resolved_db_path)
    try:
        if full_seed is not None:
            full_seed_path = Path(full_seed)
            if full_seed_path.exists():
                conn.executescript(full_seed_path.read_text(encoding="utf-8"))
                conn.commit()

        if deltas_dir is not None:
            delta_root = Path(deltas_dir)
            if delta_root.exists():
                for delta_file in sorted(delta_root.glob("*.sql")):
                    conn.executescript(delta_file.read_text(encoding="utf-8"))
                    conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", dest="db_path", type=str, default=str(default_db_path()))
    parser.add_argument(
        "--full-seed",
        dest="full_seed",
        type=str,
        default=str(DEFAULT_FULL_SEED_PATH),
        help="Pass empty string to skip full seed",
    )
    parser.add_argument(
        "--deltas-dir",
        dest="deltas_dir",
        type=str,
        default=str(DEFAULT_DELTA_SEEDS_DIR),
        help="Pass empty string to skip delta seeds",
    )
    parser.add_argument(
        "--drop-existing",
        dest="drop_existing",
        action="store_true",
    )
    args = parser.parse_args()

    full_seed = args.full_seed if args.full_seed else None
    deltas_dir = args.deltas_dir if args.deltas_dir else None
    rebuild_database(
        db_path=args.db_path,
        full_seed=full_seed,
        deltas_dir=deltas_dir,
        drop_existing=args.drop_existing,
    )
    print(f"Rebuilt DB at: {Path(args.db_path).resolve()}")


if __name__ == "__main__":
    main()
