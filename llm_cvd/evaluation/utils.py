"""Shared helpers for API evaluation runners."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable


RESULT_FIELDS = [
    "sample_id",
    "db_name",
    "provider",
    "model",
    "routed_provider",
    "routed_model",
    "fallback_attempts",
    "k",
    "repeat_id",
    "true_label",
    "pred_label",
    "raw_response",
    "is_correct",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_sec",
    "prompt_char_len",
    "error",
]


def parse_model_overrides(raw: str) -> dict[str, str]:
    overrides = {}
    if not raw:
        return overrides
    for pair in raw.split(","):
        if not pair.strip():
            continue
        provider, model = pair.split("=", 1)
        overrides[provider.strip()] = model.strip()
    return overrides


def provider_alias_key(alias: str) -> str:
    lower = alias.lower()
    if lower in {"openai", "gpt"}:
        return "chatgpt"
    if lower in {"freellmapi", "free"}:
        return "freellm"
    if lower == "google":
        return "gemini"
    if lower == "xai":
        return "grok"
    return lower


def load_completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            completed.add(
                (
                    str(row["sample_id"]),
                    row["provider"],
                    row["model"],
                    int(row.get("repeat_id", 0)),
                )
            )
    return completed


def load_result_key_sets(path: Path) -> tuple[set[tuple[str, str, str, int]], set[tuple[str, str, str, int]]]:
    """Return successful and errored result keys from a JSONL or CSV result file."""
    successful: set[tuple[str, str, str, int]] = set()
    errored: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    def add_row(row: dict[str, object]) -> None:
        key = (
            str(row["sample_id"]),
            str(row["provider"]),
            str(row["model"]),
            int(row.get("repeat_id", 0) or 0),
        )
        if row.get("error"):
            errored.add(key)
        else:
            successful.add(key)

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    add_row(json.loads(line))
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                add_row(row)

    return successful, errored


def ensure_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def append_record(jsonl_path: Path, csv_path: Path, record: dict[str, object]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    csv_fields = RESULT_FIELDS
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle), None)
            if existing_fields:
                csv_fields = existing_fields

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


def log_step(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


class timed_step:
    def __init__(self, message: str) -> None:
        self.message = message
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        log_step(f"START {self.message}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        elapsed = format_seconds(time.perf_counter() - self.started)
        if exc_type:
            log_step(f"FAIL {self.message} after {elapsed}")
        else:
            log_step(f"DONE {self.message} in {elapsed}")


def load_env_file(path: str) -> None:
    """Load .env values without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ModuleNotFoundError:
        pass

    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def progress(iterator: Iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterator, desc=desc)
    except ModuleNotFoundError:
        return iterator
