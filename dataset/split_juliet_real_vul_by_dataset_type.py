#!/usr/bin/env python3
"""Split juliet_Real_Vul_data.csv by dataset_type."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_cvd.data.juliet_loader import parse_real_vul_csv


FIELDNAMES = [
    "file_name",
    "unique_id",
    "target",
    "vulnerable_line_numbers",
    "project",
    "source_signature_path",
    "commit_hash",
    "dataset_type",
    "processed_func",
]


def parse_args() -> argparse.Namespace:
    dataset_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=dataset_dir / "juliet_Real_Vul_data.csv",
        help="Source CSV file to split.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=dataset_dir / "juliet_Real_Vul_train_val.csv",
        help="Output CSV path for rows where dataset_type=train_val.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=dataset_dir / "juliet_Real_Vul_test.csv",
        help="Output CSV path for rows where dataset_type=test.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    args = parse_args()
    records = parse_real_vul_csv(args.input.read_text(encoding="utf-8", errors="replace"))
    counts = Counter(record.get("dataset_type", "") for record in records)

    train_rows = [record for record in records if record.get("dataset_type") == "train_val"]
    test_rows = [record for record in records if record.get("dataset_type") == "test"]

    write_csv(args.train_output, train_rows)
    write_csv(args.test_output, test_rows)

    print(f"Input: {args.input}")
    print(f"Total rows: {len(records)}")
    print(f"dataset_type counts: {dict(sorted(counts.items()))}")
    print(f"Wrote {len(train_rows)} rows to {args.train_output}")
    print(f"Wrote {len(test_rows)} rows to {args.test_output}")


if __name__ == "__main__":
    main()
