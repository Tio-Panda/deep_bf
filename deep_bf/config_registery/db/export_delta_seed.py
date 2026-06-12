import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from deep_bf.config_registery.db.seed_export import (  # type: ignore
        DEFAULT_DELTA_SEEDS_DIR,
        default_seed_db_path,
        export_delta_seed,
    )
else:
    from .seed_export import (  # pragma: no cover
        DEFAULT_DELTA_SEEDS_DIR,
        default_seed_db_path,
        export_delta_seed,
    )


def _parse_pk(raw: str) -> int | dict[str, str]:
    if "=" in raw:
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        result: dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                raise ValueError(
                    "Composite PK format must be key=value pairs separated by commas"
                )
            key, value = part.split("=", maxsplit=1)
            result[key.strip()] = value.strip()
        return result

    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", dest="db_path", type=str, default=str(default_seed_db_path()))
    parser.add_argument("--table", dest="table_name", type=str, required=True)
    parser.add_argument(
        "--pk",
        dest="pk_value",
        type=str,
        required=True,
        help="Use a single value (e.g. 10) or composite key (e.g. model_id=2,family=BINN,pos=1)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default=str(DEFAULT_DELTA_SEEDS_DIR),
    )
    args = parser.parse_args()

    parsed_pk = _parse_pk(args.pk_value)
    out_file = export_delta_seed(
        table_name=args.table_name,
        pk_value=parsed_pk,
        db_path=args.db_path,
        out_dir=args.out_dir,
    )
    print(f"Wrote delta seed: {out_file}")


if __name__ == "__main__":
    main()
