"""Dataset loader for Juliet training data and Real_Vul_data.csv tests."""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_JULIET_ROOT = Path("/home/dayoung/juliet-playground/juliet-test-suite-v1.3")
DEFAULT_REAL_VUL_CSV = Path("/home/dayoung/juliet-playground/cases/Real_Vul_data.csv")
SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}


def load_juliet_experiment_data(
    juliet_root: str | Path = DEFAULT_JULIET_ROOT,
    real_vul_csv: str | Path = DEFAULT_REAL_VUL_CSV,
    cache_dir: str | Path | None = None,
    max_train_samples: int | None = None,
    seed: int = 42,
    rebuild_cache: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Juliet train examples and Real_Vul_data.csv test examples."""
    log_loader("Loading Juliet experiment data")
    train_records = load_juliet_train_records(
        juliet_root=juliet_root,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    original_train_count = len(train_records)
    if max_train_samples and len(train_records) > max_train_samples:
        log_loader(
            f"Sampling {max_train_samples} train records from {len(train_records)} "
            f"with seed={seed}"
        )
        rng = random.Random(seed)
        train_records = train_records[:]
        rng.shuffle(train_records)
        train_records = train_records[:max_train_samples]

    test_records = load_real_vul_records(real_vul_csv)
    log_loader(
        f"Prepared {len(train_records)} train records "
        f"(from {original_train_count}) and {len(test_records)} test records"
    )
    return train_records, test_records


def load_real_vul_records(path: str | Path) -> list[dict[str, Any]]:
    """Load Real_Vul_data.csv using a parser tolerant of raw quotes inside code."""
    csv_path = Path(path)
    log_loader(f"Loading Real_Vul CSV: {csv_path}")
    records = parse_real_vul_csv(csv_path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for record in records:
        code = record.get("processed_func", "").strip()
        target = record.get("target", "").strip()
        if not code or target not in {"0", "1"}:
            continue
        rows.append(
            {
                "sample_id": record.get("unique_id") or record.get("file_name") or len(rows),
                "code": code,
                "label": int(target),
                "label_text": "Vulnerable" if target == "1" else "Safe",
                "db_name": "real-vul",
                "project": record.get("project", ""),
                "source": str(csv_path),
            }
        )
    log_loader(f"Loaded {len(rows)} Real_Vul records")
    return rows


def parse_real_vul_csv(text: str) -> list[dict[str, str]]:
    """Parse the specific Real_Vul_data.csv format.

    The final processed_func column is quoted but contains raw C/C++ quotes in
    some rows, which can confuse standard CSV readers. Records begin with
    file_name,unique_id,target where the first two fields are numeric and target
    is 0/1, so we split records by that prefix and parse only the fixed prefix
    columns with CSV-like quote handling.
    """
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0].split(",")
    expected_real_vul_header = [
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
    if header[:9] != expected_real_vul_header:
        if {"file_name", "unique_id", "target", "processed_func"}.issubset(header):
            return parse_standard_real_vul_csv(text)
        raise ValueError("Unexpected Real_Vul_data.csv header.")

    starts = []
    for index, line in enumerate(lines[1:], start=1):
        if re.match(r"^\d+,\d+,[01],", line):
            starts.append(index)
    starts.append(len(lines))

    records = []
    for current, next_start in zip(starts, starts[1:]):
        raw_record = "\n".join(lines[current:next_start])
        prefix, code = split_fixed_prefix(raw_record, expected_commas=8)
        values = parse_prefix_values(prefix)
        if len(values) != 8:
            continue
        code = code.strip()
        if code.startswith('"'):
            code = code[1:]
        if code.endswith('"'):
            code = code[:-1]
        row = dict(zip(header[:8], values))
        row["processed_func"] = code
        records.append(row)
    return records


def parse_standard_real_vul_csv(text: str) -> list[dict[str, str]]:
    """Parse Real_Vul-style CSV files that are valid RFC CSV."""
    set_csv_field_size_limit()
    reader = csv.DictReader(text.splitlines())
    return [
        {str(key): value or "" for key, value in row.items() if key is not None}
        for row in reader
    ]


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def split_fixed_prefix(record: str, expected_commas: int) -> tuple[str, str]:
    in_quote = False
    comma_count = 0
    index = 0
    while index < len(record):
        char = record[index]
        if char == '"':
            if in_quote and index + 1 < len(record) and record[index + 1] == '"':
                index += 2
                continue
            in_quote = not in_quote
        elif char == "," and not in_quote:
            comma_count += 1
            if comma_count == expected_commas:
                return record[:index], record[index + 1 :]
        index += 1
    raise ValueError("Could not split Real_Vul_data.csv record prefix.")


def parse_prefix_values(prefix: str) -> list[str]:
    values = []
    current = []
    in_quote = False
    index = 0
    while index < len(prefix):
        char = prefix[index]
        if char == '"':
            if in_quote and index + 1 < len(prefix) and prefix[index + 1] == '"':
                current.append('"')
                index += 2
                continue
            in_quote = not in_quote
        elif char == "," and not in_quote:
            values.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    values.append("".join(current))
    return values


def load_juliet_train_records(
    juliet_root: str | Path,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
) -> list[dict[str, Any]]:
    root = Path(juliet_root)
    if not root.exists():
        raise FileNotFoundError(f"Juliet train source does not exist: {root}")

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / get_train_cache_name(root)
        if cache_path.exists() and not rebuild_cache:
            started = time.perf_counter()
            log_loader(f"Reading Juliet train cache: {cache_path}")
            records = read_jsonl(cache_path)
            if not records:
                log_loader(f"Cache {cache_path} is empty; rebuilding from source")
            else:
                log_loader(
                    f"Read {len(records)} Juliet train records from cache "
                    f"in {format_seconds(time.perf_counter() - started)}"
                )
                return records

    if root.is_file():
        records = load_juliet_csv_train_records(root)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            log_loader(f"Writing Juliet train cache: {cache_path}")
            write_jsonl(cache_path, records)
        return records

    records = []
    testcase_root = root / "C" / "testcases"
    source_paths = [
        source_path
        for source_path in sorted(testcase_root.rglob("*"))
        if source_path.suffix.lower() in SOURCE_EXTENSIONS
    ]
    log_loader(f"Scanning {len(source_paths)} Juliet source files under {testcase_root}")
    for source_path in progress(source_paths, desc="Loading Juliet functions"):
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        records.extend(extract_labeled_functions(text, source_path, testcase_root))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        log_loader(f"Writing Juliet train cache: {cache_path}")
        write_jsonl(cache_path, records)
    log_loader(f"Loaded {len(records)} Juliet train function records")
    return records


def get_train_cache_name(source: Path) -> str:
    if source.is_file():
        return f"{source.stem}_train_records.jsonl"
    return "juliet_train_functions.jsonl"


def load_juliet_csv_train_records(path: Path) -> list[dict[str, Any]]:
    """Load preprocessed Juliet train records from a Real_Vul-style CSV file."""
    log_loader(f"Loading Juliet train CSV: {path}")
    records = parse_real_vul_csv(path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for record in records:
        code = record.get("processed_func", "").strip()
        target = record.get("target", "").strip()
        if not code or target not in {"0", "1"}:
            continue
        unique_id = record.get("unique_id") or len(rows)
        rows.append(
            {
                "sample_id": f"juliet-csv::{unique_id}",
                "code": code,
                "label": int(target),
                "label_text": "Vulnerable" if target == "1" else "Safe",
                "db_name": "juliet",
                "project": record.get("project", ""),
                "dataset_type": record.get("dataset_type", ""),
                "source": str(path),
            }
        )
    log_loader(f"Loaded {len(rows)} Juliet CSV train records")
    return rows


def extract_labeled_functions(
    text: str,
    source_path: Path,
    testcase_root: Path,
) -> list[dict[str, Any]]:
    records = []
    function_pattern = r"^[ \t]*([A-Za-z_][\w:<>\t *&~,]*?(?:good|bad)[\w:<>\t *&~,]*)\([^;{}]*\)\s*(?:const\s*)?\{"
    for match in re.finditer(function_pattern, text, flags=re.MULTILINE):
        signature = normalize_space(match.group(0)[:-1])
        lowered = signature.lower()
        if "good" in lowered:
            label = 0
        elif "bad" in lowered:
            label = 1
        else:
            continue
        body_end = find_matching_brace(text, match.end() - 1)
        if body_end is None:
            continue
        code = text[match.start() : body_end + 1].strip()
        if len(code.splitlines()) < 3:
            continue
        relative_path = source_path.relative_to(testcase_root)
        records.append(
            {
                "sample_id": f"juliet::{relative_path}::{len(records)}",
                "code": code,
                "label": label,
                "label_text": "Vulnerable" if label == 1 else "Safe",
                "db_name": "juliet",
                "cwe": relative_path.parts[0] if relative_path.parts else "",
                "source": str(source_path),
                "signature": signature,
            }
        )
    return records


def find_matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    escaped = False

    index = open_index
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
        elif in_block_comment:
            if char == "*" and nxt == "/":
                in_block_comment = False
                index += 1
        elif in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                in_char = False
        else:
            if char == "/" and nxt == "/":
                in_line_comment = True
                index += 1
            elif char == "/" and nxt == "*":
                in_block_comment = True
                index += 1
            elif char == '"':
                in_string = True
            elif char == "'":
                in_char = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        index += 1
    return None


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_loader(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [data] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def progress(items, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ModuleNotFoundError:
        return items
