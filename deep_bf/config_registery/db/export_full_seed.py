import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from deep_bf.config_registery.db.seed_export import (  # type: ignore
        DEFAULT_FULL_SEED_PATH,
        default_seed_db_path,
        export_full_seed,
    )
else:
    from .seed_export import (  # pragma: no cover
        DEFAULT_FULL_SEED_PATH,
        default_seed_db_path,
        export_full_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", dest="db_path", type=str, default=str(default_seed_db_path()))
    parser.add_argument(
        "--out",
        dest="out_file",
        type=str,
        default=str(DEFAULT_FULL_SEED_PATH),
    )
    args = parser.parse_args()

    out_path = export_full_seed(db_path=args.db_path, out_file=args.out_file)
    print(f"Wrote full seed: {out_path}")


if __name__ == "__main__":
    main()
