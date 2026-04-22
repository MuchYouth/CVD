"""Run API-based few-shot RAG evaluation and save every provider result immediately."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from prompting import SYSTEM_PROMPT, build_few_shot_prompt, parse_label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RESULT_FIELDS = [
    "sample_id",
    "db_name",
    "provider",
    "model",
    "k",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="juliet-real")
    parser.add_argument(
        "--rag-dataset-root",
        "--juliet-root",
        dest="rag_dataset_root",
        default="../../juliet-playground/juliet-test-suite-v1.3",
        help="Root directory for the RAG training dataset.",
    )
    parser.add_argument(
        "--target-dataset-csv",
        "--rag-test-csv",
        "--real-vul-csv",
        dest="target_dataset_csv",
        default="../../juliet-playground/cases/Real_Vul_data.csv",
        help="CSV file for the target dataset classified in each prompt.",
    )
    parser.add_argument("--providers", default="chatgpt,claude,gemini,grok")
    parser.add_argument("--models", default="", help="Optional provider=model pairs, comma-separated.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_train_codebert")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    load_env_file(args.env_file)

    from juliet_loader import load_juliet_experiment_data
    from providers import make_client, resolve_provider_model
    from rag_retriever import CodeBertFaissRetriever

    provider_aliases = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    model_overrides = parse_model_overrides(args.models)

    log_step("Starting RAG API evaluation")
    log_step(f"Provider aliases: {', '.join(provider_aliases)}")
    with timed_step("Load Juliet train and target dataset"):
        train_records, test_records = load_juliet_experiment_data(
            juliet_root=args.rag_dataset_root,
            real_vul_csv=args.target_dataset_csv,
            cache_dir=args.cache_dir,
            max_train_samples=args.max_train_samples,
            seed=args.seed,
            rebuild_cache=args.rebuild_cache,
        )
    log_step(f"Loaded {len(train_records)} train records and {len(test_records)} test records")

    index_name = args.index_name
    if args.max_train_samples:
        index_name = f"{index_name}_n{args.max_train_samples}_seed{args.seed}"
    with timed_step(f"Load or build FAISS index '{index_name}'"):
        retriever = CodeBertFaissRetriever(
            train_records=train_records,
            index_name=index_name,
            index_dir=Path(args.index_dir),
            rebuild_index=args.rebuild_index,
        )
    clients = []
    with timed_step("Initialize API clients"):
        for alias in provider_aliases:
            provider, model = resolve_provider_model(
                alias,
                model_override=model_overrides.get(alias) or model_overrides.get(provider_alias_key(alias)),
            )
            client = make_client(provider, model=model)
            clients.append(client)
            log_step(f"Client ready: alias={alias}, provider={client.provider}, model={client.model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.db_name}_k{args.k}"
    jsonl_path = output_dir / f"{run_name}.jsonl"
    csv_path = output_dir / f"{run_name}.csv"

    completed = load_completed_keys(jsonl_path) if args.resume else set()
    ensure_csv_header(csv_path)
    if completed:
        log_step(f"Resume enabled: loaded {len(completed)} completed result keys")
    log_step(f"Writing results to {jsonl_path} and {csv_path}")

    stop = len(test_records) if args.limit is None else min(len(test_records), args.start + args.limit)
    iterator = range(args.start, stop)
    total_expected = len(iterator) * len(clients)
    skipped = 0
    written = 0
    log_step(
        f"Evaluating samples {args.start}..{stop - 1} "
        f"({len(iterator)} samples x {len(clients)} clients = {total_expected} attempts)"
    )

    for sample_id in progress(iterator, desc="Evaluating samples"):
        sample_started = time.perf_counter()
        row = test_records[sample_id]
        output_sample_id = str(row.get("sample_id", sample_id))
        true_label = str(row["label_text"])
        with timed_step(f"Retrieve top-{args.k} examples for sample {output_sample_id}"):
            examples = retriever.retrieve(str(row["code"]), k=args.k)
            prompt = build_few_shot_prompt(examples, str(row["code"]))

        for client in clients:
            key = (output_sample_id, client.provider, client.model)
            if key in completed:
                skipped += 1
                continue

            log_step(f"Calling {client.provider}/{client.model} for sample {output_sample_id}")
            result = client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
            pred_label = parse_label(result.text)
            record = {
                "sample_id": output_sample_id,
                "db_name": args.db_name,
                "provider": result.provider,
                "model": result.model,
                "k": args.k,
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_response": result.text,
                "is_correct": bool(pred_label == true_label) if pred_label else False,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "latency_sec": round(result.latency_sec, 6),
                "prompt_char_len": len(prompt),
                "error": result.error,
            }
            append_record(jsonl_path, csv_path, record)
            completed.add(key)
            written += 1
            status = "error" if result.error else f"pred={pred_label}"
            log_step(
                f"Saved {client.provider}/{client.model} sample {output_sample_id}: "
                f"{status}, latency={result.latency_sec:.3f}s"
            )
        log_step(f"Finished sample {output_sample_id} in {format_seconds(time.perf_counter() - sample_started)}")

    log_step(
        f"Done. wrote={written}, skipped={skipped}, "
        f"elapsed={format_seconds(time.perf_counter() - run_started)}"
    )


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
    if lower == "google":
        return "gemini"
    if lower == "xai":
        return "grok"
    return lower


def load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((str(row["sample_id"]), row["provider"], row["model"]))
    return completed


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

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
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


def progress(iterator, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterator, desc=desc)
    except ModuleNotFoundError:
        return iterator


if __name__ == "__main__":
    main()
