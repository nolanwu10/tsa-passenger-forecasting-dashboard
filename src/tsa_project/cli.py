import argparse
import json
from typing import Sequence

from tsa_project.datasets import inventory, read_csv_dataset, summarize_dataframe
from tsa_project.quality import validate_dataset
from tsa_project.schemas import DATASET_SPECS


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def handle_inventory() -> int:
    _print_json(inventory())
    return 0


def handle_profile(dataset: str) -> int:
    df = read_csv_dataset(dataset)
    _print_json(summarize_dataframe(df))
    return 0


def handle_validate(dataset: str) -> int:
    df = read_csv_dataset(dataset)
    spec = DATASET_SPECS[dataset]
    errors = validate_dataset(df, spec)
    if errors:
        _print_json({"dataset": dataset, "valid": False, "errors": errors})
        return 1

    _print_json({"dataset": dataset, "valid": True, "errors": []})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect TSA project datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory", help="List known datasets and whether they exist.")

    profile_parser = subparsers.add_parser("profile", help="Summarize a dataset.")
    profile_parser.add_argument("dataset", choices=sorted(DATASET_SPECS))

    validate_parser = subparsers.add_parser("validate", help="Validate a dataset contract.")
    validate_parser.add_argument("dataset", choices=sorted(DATASET_SPECS))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "inventory":
        return handle_inventory()
    if args.command == "profile":
        return handle_profile(args.dataset)
    if args.command == "validate":
        return handle_validate(args.dataset)

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

