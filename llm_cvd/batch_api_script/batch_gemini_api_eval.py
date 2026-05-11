"""Create and collect Gemini Batch API jobs for zero-shot or RAG CVD evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_cvd.evaluation.utils import (
    append_record,
    ensure_csv_header,
    format_seconds,
    load_env_file,
    log_step,
    timed_step,
)
from llm_cvd.prompts.templates import build_few_shot_prompt, build_zero_shot_prompt, parse_label


BATCH_SYSTEM_PROMPT = (
    "You are a binary vulnerability classifier. "
    "Classify the given source code as Vulnerable or Safe. "
    "Return only JSON that matches the requested schema."
)

LABEL_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": ["Vulnerable", "Safe"]},
    },
    "required": ["label"],
}

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(common_parser: argparse.ArgumentParser) -> None:
        common_parser.add_argument("--env-file", default=".env")
        common_parser.add_argument("--output-dir", default="results")
        common_parser.add_argument("--run-name", default=None)
        common_parser.add_argument("--db-name", default="juliet-real")
        common_parser.add_argument("--model", default=None)

    def add_dataset(dataset_parser: argparse.ArgumentParser) -> None:
        dataset_parser.add_argument(
            "--target-dataset-csv",
            "--rag-test-csv",
            "--real-vul-csv",
            dest="target_dataset_csv",
            default="../../juliet-playground/cases/Real_Vul_data.csv",
            help="CSV file for the target dataset classified in each prompt.",
        )
        dataset_parser.add_argument("--start", type=int, default=0)
        dataset_parser.add_argument("--limit", type=int, default=None)
        dataset_parser.add_argument("--repeats", type=int, default=1)

    def add_prompt_options(prompt_parser: argparse.ArgumentParser) -> None:
        prompt_parser.add_argument(
            "--prompt-mode",
            choices=["zero-shot", "rag"],
            default="zero-shot",
            help="Use zero-shot prompts, or retrieve top-k examples and build RAG few-shot prompts.",
        )
        prompt_parser.add_argument(
            "--rag-dataset-root",
            "--juliet-root",
            dest="rag_dataset_root",
            default="../../juliet-playground/juliet-test-suite-v1.3",
            help="Root directory for raw Juliet testcases, or a Real_Vul-style CSV for RAG training.",
        )
        prompt_parser.add_argument("--k", type=int, default=6)
        prompt_parser.add_argument("--cache-dir", default="cache")
        prompt_parser.add_argument("--index-dir", default="indexes")
        prompt_parser.add_argument("--index-name", default="juliet_train_codebert")
        prompt_parser.add_argument("--retrieval-batch-size", type=int, default=16)
        prompt_parser.add_argument("--max-train-samples", type=int, default=None)
        prompt_parser.add_argument("--seed", type=int, default=42)
        prompt_parser.add_argument("--rebuild-cache", action="store_true")
        prompt_parser.add_argument("--rebuild-index", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Write Gemini Batch API input JSONL and manifest files.")
    add_common(prepare)
    add_dataset(prepare)
    add_prompt_options(prepare)
    prepare.add_argument("--max-output-tokens", type=int, default=64)
    prepare.add_argument("--temperature", type=float, default=0.0)
    prepare.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help=(
            "Thinking budget for Gemini 2.5 models. "
            "Use -1 to omit thinkingConfig. Budget 0 is skipped for 2.5 Pro."
        ),
    )

    submit = subparsers.add_parser("submit", help="Upload the prepared JSONL file and create a batch job.")
    add_common(submit)
    submit.add_argument("--input-jsonl", default=None)

    status = subparsers.add_parser("status", help="Fetch the current state of a batch job.")
    add_common(status)
    status.add_argument("--job-name", "--batch-id", dest="job_name", default=None)
    status.add_argument("--job-file", default=None)

    collect = subparsers.add_parser("collect", help="Download and parse a completed batch result file.")
    add_common(collect)
    collect.add_argument("--job-name", "--batch-id", dest="job_name", default=None)
    collect.add_argument("--job-file", default=None)
    collect.add_argument("--manifest-jsonl", default=None)
    collect.add_argument("--result-jsonl", default=None)

    run = subparsers.add_parser("run", help="Prepare, submit, poll, and collect in one command.")
    add_common(run)
    add_dataset(run)
    add_prompt_options(run)
    run.add_argument("--max-output-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help=(
            "Thinking budget for Gemini 2.5 models. "
            "Use -1 to omit thinkingConfig. Budget 0 is skipped for 2.5 Pro."
        ),
    )
    run.add_argument("--poll-interval-sec", type=int, default=60)

    return parser.parse_args()


def get_prompt_mode(args: argparse.Namespace) -> str:
    return getattr(args, "prompt_mode", "zero-shot")


def get_result_k(args: argparse.Namespace) -> int:
    return int(args.k) if get_prompt_mode(args) == "rag" else 0


def get_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    mode = get_prompt_mode(args)
    if mode == "rag":
        return f"{args.db_name}_k{args.k}_gemini_batch"
    return f"{args.db_name}_gemini_batch"


def get_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    run_name = get_run_name(args)
    return {
        "output_dir": output_dir,
        "input_jsonl": output_dir / f"batch_gemini_{run_name}_requests.jsonl",
        "manifest_jsonl": output_dir / f"batch_gemini_{run_name}_manifest.jsonl",
        "job_json": output_dir / f"batch_gemini_{run_name}_job.json",
        "raw_result_jsonl": output_dir / f"batch_gemini_{run_name}_raw_results.jsonl",
        "result_jsonl": output_dir / f"{run_name}.jsonl",
        "result_csv": output_dir / f"{run_name}.csv",
        "parse_fails_json": output_dir / f"batch_gemini_{run_name}_parse_fails.json",
    }


def require_google_genai():
    try:
        from google import genai
        from google.genai import types
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Gemini Batch API uses the google-genai package. "
            "Install it in the Python environment you are using, for example: "
            "python3 -m pip install google-genai"
        ) from exc
    return genai, types


def make_client():
    genai, _ = require_google_genai()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before using the Gemini Batch API.")
    return genai.Client(api_key=api_key)


def make_generation_config(
    model: str,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": LABEL_RESPONSE_SCHEMA,
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if model.startswith("gemini-2.5") and thinking_budget >= 0:
        if thinking_budget > 0 or "pro" not in model.lower():
            config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return config


def make_batch_request(
    prompt: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int,
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": BATCH_SYSTEM_PROMPT}],
        },
        "generationConfig": make_generation_config(model, max_output_tokens, temperature, thinking_budget),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["key"])] = row
    return rows


def resolve_rag_index_name(args: argparse.Namespace) -> str:
    index_name = args.index_name
    rag_dataset_path = Path(args.rag_dataset_root)
    if index_name == "juliet_train_codebert" and rag_dataset_path.is_file():
        index_name = f"{rag_dataset_path.stem}_codebert"
    if args.max_train_samples:
        index_name = f"{index_name}_n{args.max_train_samples}_seed{args.seed}"
    return index_name


def load_records_and_prompts(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]], list[str]]:
    mode = get_prompt_mode(args)
    if mode == "zero-shot":
        from llm_cvd.data.juliet_loader import load_real_vul_records

        with timed_step("Load target dataset"):
            records = load_real_vul_records(args.target_dataset_csv)

        stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
        selected_indexes = list(range(args.start, stop))
        selected_rows = [records[index] for index in selected_indexes]
        prompts = [build_zero_shot_prompt(str(row["code"])) for row in selected_rows]
        return selected_indexes, selected_rows, prompts

    if mode != "rag":
        raise ValueError(f"Unknown prompt mode: {mode}")

    if mode == "rag":
        if args.k < 1:
            raise ValueError("--k must be at least 1 when --prompt-mode rag")
        if args.retrieval_batch_size < 1:
            raise ValueError("--retrieval-batch-size must be at least 1")
        from llm_cvd.data.juliet_loader import load_juliet_experiment_data
        try:
            from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RAG mode requires the retrieval dependencies from requirements.txt "
                "(notably faiss-cpu, torch, and transformers). Install them in the "
                "Python environment you are using before running --prompt-mode rag."
            ) from exc

        with timed_step("Load Juliet train and target dataset"):
            train_records, test_records = load_juliet_experiment_data(
                juliet_root=args.rag_dataset_root,
                real_vul_csv=args.target_dataset_csv,
                cache_dir=args.cache_dir,
                max_train_samples=args.max_train_samples,
                seed=args.seed,
                rebuild_cache=args.rebuild_cache,
            )
        log_step(f"Loaded {len(train_records)} train records and {len(test_records)} target records")

        index_name = resolve_rag_index_name(args)
        with timed_step(f"Load or build FAISS index '{index_name}'"):
            retriever = CodeBertFaissRetriever(
                train_records=train_records,
                index_name=index_name,
                index_dir=Path(args.index_dir),
                rebuild_index=args.rebuild_index,
            )

    stop = len(test_records) if args.limit is None else min(len(test_records), args.start + args.limit)
    selected_indexes = list(range(args.start, stop))
    selected_rows = [test_records[index] for index in selected_indexes]

    with timed_step(
        f"Retrieve top-{args.k} examples for {len(selected_rows)} samples "
        f"(batch_size={args.retrieval_batch_size})"
    ):
        retrieved_examples = retriever.retrieve_many(
            [str(row["code"]) for row in selected_rows],
            k=args.k,
            batch_size=args.retrieval_batch_size,
            desc="Embedding target records",
        )
        prompts = [
            build_few_shot_prompt(examples, str(row["code"]))
            for row, examples in zip(selected_rows, retrieved_examples)
        ]

    return selected_indexes, selected_rows, prompts


def prepare_batch(args: argparse.Namespace) -> dict[str, Path]:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.start < 0:
        raise ValueError("--start must be non-negative")

    paths = get_paths(args)
    selected_indexes, selected_rows, prompts = load_records_and_prompts(args)
    requests = []
    manifest = []
    result_k = get_result_k(args)

    for sample_index, row, prompt in zip(selected_indexes, selected_rows, prompts):
        output_sample_id = str(row.get("sample_id", sample_index))
        for repeat_id in range(args.repeats):
            key = f"request-{len(requests)}"
            requests.append(
                {
                    "key": key,
                    "request": make_batch_request(
                        prompt=prompt,
                        model=args.model,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        thinking_budget=args.thinking_budget,
                    ),
                }
            )
            manifest.append(
                {
                    "key": key,
                    "sample_id": output_sample_id,
                    "sample_index": sample_index,
                    "db_name": args.db_name,
                    "provider": "gemini",
                    "model": args.model,
                    "prompt_mode": get_prompt_mode(args),
                    "k": result_k,
                    "repeat_id": repeat_id,
                    "true_label": str(row["label_text"]),
                    "prompt_char_len": len(prompt),
                }
            )

    write_jsonl(paths["input_jsonl"], requests)
    write_jsonl(paths["manifest_jsonl"], manifest)
    log_step(f"Prepared {get_prompt_mode(args)} prompts with k={result_k}")
    log_step(f"Wrote {len(requests)} batch requests to {paths['input_jsonl']}")
    log_step(f"Wrote manifest to {paths['manifest_jsonl']}")
    return paths


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    _, types = require_google_genai()
    client = make_client()
    paths = get_paths(args)
    input_jsonl_arg = getattr(args, "input_jsonl", None)
    input_jsonl = Path(input_jsonl_arg) if input_jsonl_arg else paths["input_jsonl"]
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Batch input JSONL not found: {input_jsonl}")

    with timed_step(f"Upload {input_jsonl}"):
        uploaded_file = client.files.upload(
            file=str(input_jsonl),
            config=types.UploadFileConfig(
                display_name=input_jsonl.stem,
                mime_type="jsonl",
            ),
        )
    with timed_step("Create Gemini batch job"):
        batch_job = client.batches.create(
            model=args.model,
            src=uploaded_file.name,
            config={"display_name": f"batch-gemini-{get_run_name(args)}"},
        )

    job_info = {
        "name": batch_job.name,
        "model": args.model,
        "uploaded_file": uploaded_file.name,
        "input_jsonl": str(input_jsonl),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    paths["job_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["job_json"].write_text(json.dumps(job_info, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"Created batch job: {batch_job.name}")
    log_step(f"Wrote job metadata to {paths['job_json']}")
    return job_info


def load_job_name(args: argparse.Namespace) -> str:
    job_name = getattr(args, "job_name", None)
    if job_name:
        return job_name
    paths = get_paths(args)
    job_file_arg = getattr(args, "job_file", None)
    job_file = Path(job_file_arg) if job_file_arg else paths["job_json"]
    if not job_file.exists():
        raise FileNotFoundError(f"Job metadata file not found: {job_file}")
    return str(json.loads(job_file.read_text(encoding="utf-8"))["name"])


def get_batch_job(args: argparse.Namespace):
    client = make_client()
    job_name = load_job_name(args)
    return client.batches.get(name=job_name)


def status_batch(args: argparse.Namespace):
    batch_job = get_batch_job(args)
    state = batch_job.state.name
    log_step(f"Batch job: {batch_job.name}")
    log_step(f"Current state: {state}")
    if state == "JOB_STATE_FAILED":
        log_step(f"Error: {batch_job.error}")
    return batch_job


def extract_text_from_response(result: dict[str, Any]) -> str | None:
    if "error" in result:
        return None
    response = result.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return None
    return parts[0].get("text")


def parse_pred_label(text: str | None) -> str | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            label = parsed.get("label")
            if label in {"Vulnerable", "Safe"}:
                return str(label)
    except json.JSONDecodeError:
        pass
    return parse_label(text)


def extract_usage(result: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = (result.get("response") or {}).get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount")
    total_tokens = usage.get("totalTokenCount")
    output_tokens = None
    if input_tokens is not None and total_tokens is not None:
        output_tokens = total_tokens - input_tokens
    return input_tokens, output_tokens, total_tokens


def describe_missing_text(result: dict[str, Any]) -> str:
    response = result.get("response") or {}
    candidates = response.get("candidates") or []
    finish_reason = candidates[0].get("finishReason") if candidates else None
    usage = response.get("usageMetadata") or {}
    details = []
    if finish_reason:
        details.append(f"finishReason={finish_reason}")
    for key in ("thoughtsTokenCount", "promptTokenCount", "totalTokenCount"):
        if key in usage:
            details.append(f"{key}={usage[key]}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Missing response candidate text{suffix}"


def make_result_record(meta: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_response = extract_text_from_response(result)
    pred_label = parse_pred_label(raw_response)
    input_tokens, output_tokens, total_tokens = extract_usage(result)
    error = None
    parse_fail = None

    if "error" in result:
        error = json.dumps(result["error"], ensure_ascii=False)
        parse_fail = {"key": meta["key"], "reason": "request_error", "response": result}
    elif raw_response is None:
        error = describe_missing_text(result)
        parse_fail = {"key": meta["key"], "reason": "missing_response_text", "response": result}
    elif pred_label is None:
        error = "Could not parse label"
        parse_fail = {"key": meta["key"], "reason": "parse_error", "raw_response": raw_response}

    true_label = str(meta["true_label"])
    record = {
        "sample_id": meta["sample_id"],
        "db_name": meta["db_name"],
        "provider": "gemini",
        "model": meta["model"],
        "k": int(meta.get("k", 0)),
        "repeat_id": int(meta.get("repeat_id", 0)),
        "true_label": true_label,
        "pred_label": pred_label,
        "raw_response": raw_response,
        "is_correct": bool(pred_label == true_label) if pred_label else False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": None,
        "prompt_char_len": int(meta.get("prompt_char_len", 0)),
        "error": error,
    }
    return record, parse_fail


def collect_batch(args: argparse.Namespace) -> None:
    client = make_client()
    paths = get_paths(args)
    manifest_jsonl_arg = getattr(args, "manifest_jsonl", None)
    manifest_jsonl = Path(manifest_jsonl_arg) if manifest_jsonl_arg else paths["manifest_jsonl"]
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"Manifest JSONL not found: {manifest_jsonl}")
    manifest = read_jsonl_map(manifest_jsonl)

    result_jsonl_arg = getattr(args, "result_jsonl", None)
    result_jsonl = Path(result_jsonl_arg) if result_jsonl_arg else paths["raw_result_jsonl"]
    batch_job = get_batch_job(args)
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job is not complete: {batch_job.state.name}")
    redownload = bool(getattr(args, "redownload", False))
    if redownload or not result_jsonl.exists():
        if not batch_job.dest or not batch_job.dest.file_name:
            raise RuntimeError("Completed batch job does not expose a destination file.")
        with timed_step(f"Download result file {batch_job.dest.file_name}"):
            file_content = client.files.download(file=batch_job.dest.file_name)
            result_jsonl.parent.mkdir(parents=True, exist_ok=True)
            result_jsonl.write_bytes(file_content)

    ensure_csv_header(paths["result_csv"])
    parse_fails = []
    written = 0
    with result_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            key = str(result.get("key", ""))
            meta = manifest.get(key)
            if meta is None:
                parse_fails.append({"key": key, "reason": "missing_manifest", "response": result})
                continue
            record, parse_fail = make_result_record(meta, result)
            append_record(paths["result_jsonl"], paths["result_csv"], record)
            written += 1
            if parse_fail:
                parse_fails.append(parse_fail)

    paths["parse_fails_json"].write_text(
        json.dumps(parse_fails, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_step(f"Wrote {written} parsed records to {paths['result_jsonl']} and {paths['result_csv']}")
    log_step(f"Wrote {len(parse_fails)} parse failures to {paths['parse_fails_json']}")


def run_all(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    prepare_batch(args)
    submit_batch(args)
    while True:
        batch_job = status_batch(args)
        state = batch_job.state.name
        if state in TERMINAL_STATES:
            break
        log_step(f"Sleeping {args.poll_interval_sec}s before next status check")
        time.sleep(args.poll_interval_sec)
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job finished without success: {batch_job.state.name}")
    args.redownload = True
    collect_batch(args)
    log_step(f"Done in {format_seconds(time.perf_counter() - started)}")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.model is None:
        args.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if args.command == "prepare":
        prepare_batch(args)
    elif args.command == "submit":
        submit_batch(args)
    elif args.command == "status":
        status_batch(args)
    elif args.command == "collect":
        collect_batch(args)
    elif args.command == "run":
        run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
