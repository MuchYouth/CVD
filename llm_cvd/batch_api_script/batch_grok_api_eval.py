"""Create and collect xAI Grok Batch API jobs for CVD evaluation."""

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
    load_result_key_sets,
    log_step,
    timed_step,
)
from llm_cvd.prompts.templates import build_few_shot_prompt, build_zero_shot_prompt, parse_label


BATCH_SYSTEM_PROMPT = (
    "You are a binary vulnerability classifier. "
    "Classify the given source code as Vulnerable or Safe. "
    'Return only a JSON object in this exact shape: {"label":"Vulnerable"} '
    'or {"label":"Safe"}. Do not include markdown, reasoning, or extra text.'
)


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
        dataset_parser.add_argument(
            "--retry-errors-from",
            default=None,
            help=(
                "Optional CSV/JSONL result file. When set, only calls that errored in this "
                "file and do not already have a successful row in this file are prepared."
            ),
        )

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

    prepare = subparsers.add_parser("prepare", help="Write Grok Batch API input JSONL and manifest files.")
    add_common(prepare)
    add_dataset(prepare)
    add_prompt_options(prepare)
    prepare.add_argument("--max-output-tokens", type=int, default=64)
    prepare.add_argument("--temperature", type=float, default=0.0)

    submit = subparsers.add_parser("submit", help="Create a Grok batch and add prepared requests.")
    add_common(submit)
    submit.add_argument("--input-jsonl", default=None)
    submit.add_argument(
        "--submit-mode",
        choices=["inline", "file"],
        default="inline",
        help=(
            "inline uses client.batch.create() plus client.batch.add() with SDK request objects. "
            "file uploads the JSONL and creates a sealed file-based batch."
        ),
    )

    status = subparsers.add_parser("status", help="Fetch the current state of a batch job.")
    add_common(status)
    status.add_argument("--job-name", "--batch-id", dest="job_name", default=None)
    status.add_argument("--job-file", default=None)

    collect = subparsers.add_parser("collect", help="Download and parse completed Grok batch results.")
    add_common(collect)
    collect.add_argument("--job-name", "--batch-id", dest="job_name", default=None)
    collect.add_argument("--job-file", default=None)
    collect.add_argument("--manifest-jsonl", default=None)
    collect.add_argument("--result-jsonl", default=None)
    collect.add_argument("--redownload", action="store_true")
    collect.add_argument("--page-size", type=int, default=100)

    run = subparsers.add_parser("run", help="Prepare, submit, poll, and collect in one command.")
    add_common(run)
    add_dataset(run)
    add_prompt_options(run)
    run.add_argument("--max-output-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--poll-interval-sec", type=int, default=60)
    run.add_argument("--page-size", type=int, default=100)
    run.add_argument(
        "--submit-mode",
        choices=["inline", "file"],
        default="inline",
        help="Submission mode used after prepare.",
    )

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
        return f"{args.db_name}_k{args.k}_grok_batch"
    return f"{args.db_name}_grok_batch"


def get_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    run_name = get_run_name(args)
    return {
        "output_dir": output_dir,
        "input_jsonl": output_dir / f"batch_grok_{run_name}_requests.jsonl",
        "manifest_jsonl": output_dir / f"batch_grok_{run_name}_manifest.jsonl",
        "job_json": output_dir / f"batch_grok_{run_name}_job.json",
        "raw_result_jsonl": output_dir / f"batch_grok_{run_name}_raw_results.jsonl",
        "result_jsonl": output_dir / f"{run_name}.jsonl",
        "result_csv": output_dir / f"{run_name}.csv",
        "parse_fails_json": output_dir / f"batch_grok_{run_name}_parse_fails.json",
    }


def require_xai_sdk():
    try:
        from xai_sdk import Client
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Grok Batch API uses the xai-sdk package. Install it with: pip install xai-sdk"
        ) from exc
    return Client


def make_client():
    Client = require_xai_sdk()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set XAI_API_KEY before using the Grok Batch API.")
    return Client()


def make_batch_request(
    custom_id: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["key"])] = row
    return rows


def load_retry_keys(
    args: argparse.Namespace,
) -> tuple[set[tuple[str, str, str, int]] | None, set[str] | None]:
    retry_errors_from = getattr(args, "retry_errors_from", None)
    if not retry_errors_from:
        return None, None

    retry_path = Path(retry_errors_from)
    with timed_step(f"Load retry keys from {retry_path}"):
        source_successful, source_errored = load_result_key_sets(retry_path)
    retry_keys = source_errored - source_successful
    retry_sample_ids = {key[0] for key in retry_keys}
    log_step(
        f"Retry-errors mode: {len(retry_keys)} failed calls across "
        f"{len(retry_sample_ids)} samples will be prepared"
    )
    return retry_keys, retry_sample_ids


def filter_selected_indexes_for_retry(
    selected_indexes: list[int],
    rows: list[dict[str, Any]],
    retry_sample_ids: set[str] | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    if retry_sample_ids is None:
        return selected_indexes, rows
    filtered_indexes = []
    filtered_rows = []
    for sample_index, row in zip(selected_indexes, rows):
        output_sample_id = str(row.get("sample_id", sample_index))
        if output_sample_id in retry_sample_ids:
            filtered_indexes.append(sample_index)
            filtered_rows.append(row)
    return filtered_indexes, filtered_rows


def load_records_and_prompts(
    args: argparse.Namespace,
    retry_sample_ids: set[str] | None = None,
) -> tuple[list[int], list[dict[str, Any]], list[str]]:
    mode = get_prompt_mode(args)
    if mode == "zero-shot":
        from llm_cvd.data.juliet_loader import load_real_vul_records

        with timed_step("Load target dataset"):
            records = load_real_vul_records(args.target_dataset_csv)

        stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
        selected_indexes = list(range(args.start, stop))
        selected_rows = [records[index] for index in selected_indexes]
        selected_indexes, selected_rows = filter_selected_indexes_for_retry(
            selected_indexes,
            selected_rows,
            retry_sample_ids,
        )
        prompts = [build_zero_shot_prompt(str(row["code"])) for row in selected_rows]
        return selected_indexes, selected_rows, prompts

    if mode != "rag":
        raise ValueError(f"Unknown prompt mode: {mode}")

    if args.k < 1:
        raise ValueError("--k must be at least 1 when --prompt-mode rag")
    if args.retrieval_batch_size < 1:
        raise ValueError("--retrieval-batch-size must be at least 1")

    from llm_cvd.data.juliet_loader import load_juliet_experiment_data
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever

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

    index_name = args.index_name
    rag_dataset_path = Path(args.rag_dataset_root)
    if index_name == "juliet_train_codebert" and rag_dataset_path.is_file():
        index_name = f"{rag_dataset_path.stem}_codebert"
    if args.max_train_samples:
        index_name = f"{index_name}_n{args.max_train_samples}_seed{args.seed}"

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
    selected_indexes, selected_rows = filter_selected_indexes_for_retry(
        selected_indexes,
        selected_rows,
        retry_sample_ids,
    )
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
    retry_keys, retry_sample_ids = load_retry_keys(args)
    selected_indexes, selected_rows, prompts = load_records_and_prompts(args, retry_sample_ids)
    requests = []
    manifest = []
    result_k = get_result_k(args)
    skipped = 0

    for sample_index, row, prompt in zip(selected_indexes, selected_rows, prompts):
        output_sample_id = str(row.get("sample_id", sample_index))
        for repeat_id in range(args.repeats):
            result_key = (output_sample_id, "grok", str(args.model), repeat_id)
            if retry_keys is not None and result_key not in retry_keys:
                skipped += 1
                continue
            key = f"request-{len(requests)}"
            requests.append(
                make_batch_request(
                    custom_id=key,
                    model=args.model,
                    prompt=prompt,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
            )
            manifest.append(
                {
                    "key": key,
                    "sample_id": output_sample_id,
                    "sample_index": sample_index,
                    "db_name": args.db_name,
                    "provider": "grok",
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
    if retry_keys is not None:
        log_step(f"Skipped {skipped} non-error repeat/provider/model combinations")
        if not requests:
            log_step(
                "No retry requests matched this range; wrote empty request and manifest files."
            )
    log_step(f"Wrote manifest to {paths['manifest_jsonl']}")
    return paths


def get_object_id(value: Any) -> str:
    for attr in ("id", "file_id", "batch_id", "name"):
        object_id = getattr(value, attr, None)
        if object_id:
            return str(object_id)
    if isinstance(value, dict):
        for key in ("id", "file_id", "batch_id", "name"):
            object_id = value.get(key)
            if object_id:
                return str(object_id)
    raise RuntimeError(f"Could not determine object id from {value!r}")


def make_sdk_chat_request(client: Any, request: dict[str, Any]) -> Any:
    """Convert a prepared JSONL chat request into an xAI SDK batch request."""
    try:
        from xai_sdk.chat import system, user
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Grok Batch API uses xai_sdk.chat helpers. Install/update with: pip install -U xai-sdk"
        ) from exc

    custom_id = str(request["custom_id"])
    body = request.get("body") or {}
    model = str(body.get("model") or "")
    if not model:
        raise ValueError(f"Missing model for request {custom_id}")

    create_kwargs: dict[str, Any] = {
        "model": model,
        "batch_request_id": custom_id,
    }
    for key in ("temperature", "max_tokens", "max_completion_tokens"):
        if key in body and body[key] is not None:
            create_kwargs[key] = body[key]

    try:
        chat = client.chat.create(**create_kwargs)
    except TypeError:
        chat = client.chat.create(model=model, batch_request_id=custom_id)

    for message in body.get("messages", []):
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system":
            chat.append(system(content))
        elif role == "user":
            chat.append(user(content))
        else:
            raise ValueError(
                f"Unsupported role {role!r} in prepared Grok batch request {custom_id}. "
                "This evaluator only prepares system/user chat messages."
            )
    return chat


def load_sdk_batch_requests(client: Any, input_jsonl: Path) -> list[Any]:
    requests = read_jsonl(input_jsonl)
    batch_requests = []
    seen_ids: set[str] = set()
    for request in requests:
        custom_id = str(request.get("custom_id", ""))
        if not custom_id:
            raise ValueError(f"Missing custom_id in {input_jsonl}")
        if custom_id in seen_ids:
            raise ValueError(f"Duplicate custom_id in {input_jsonl}: {custom_id}")
        seen_ids.add(custom_id)
        if request.get("method") != "POST" or request.get("url") != "/v1/chat/completions":
            raise ValueError(
                f"Unsupported prepared request for {custom_id}: "
                f"{request.get('method')} {request.get('url')}. "
                "Use JSONL file submit mode for mixed endpoints."
            )
        batch_requests.append(make_sdk_chat_request(client, request))
    return batch_requests


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    client = make_client()
    paths = get_paths(args)
    input_jsonl_arg = getattr(args, "input_jsonl", None)
    input_jsonl = Path(input_jsonl_arg) if input_jsonl_arg else paths["input_jsonl"]
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Batch input JSONL not found: {input_jsonl}")

    submit_mode = getattr(args, "submit_mode", "inline")
    input_file_id = None

    if submit_mode == "file":
        with timed_step(f"Upload {input_jsonl}"):
            with input_jsonl.open("rb") as handle:
                uploaded_file = client.files.upload(file=handle)
        input_file_id = get_object_id(uploaded_file)
        with timed_step("Create sealed Grok file batch"):
            batch_job = client.batch.create(
                batch_name=f"batch-grok-{get_run_name(args)}",
                input_file_id=input_file_id,
            )
    else:
        with timed_step(f"Build SDK batch requests from {input_jsonl}"):
            batch_requests = load_sdk_batch_requests(client, input_jsonl)
        if not batch_requests:
            raise RuntimeError(f"No batch requests found in {input_jsonl}")
        with timed_step("Create Grok batch"):
            batch_job = client.batch.create(batch_name=f"batch-grok-{get_run_name(args)}")
        with timed_step(f"Add {len(batch_requests)} requests to Grok batch"):
            client.batch.add(batch_id=get_object_id(batch_job), batch_requests=batch_requests)

    batch_id = get_object_id(batch_job)

    job_info = {
        "id": batch_id,
        "name": batch_id,
        "batch_id": batch_id,
        "model": args.model,
        "input_file_id": input_file_id,
        "input_jsonl": str(input_jsonl),
        "submit_mode": submit_mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    paths["job_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["job_json"].write_text(json.dumps(job_info, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"Created batch job: {batch_id}")
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
    job_info = json.loads(job_file.read_text(encoding="utf-8"))
    return str(job_info.get("batch_id") or job_info.get("id") or job_info["name"])


def get_batch_job(args: argparse.Namespace):
    client = make_client()
    job_name = load_job_name(args)
    return client.batch.get(batch_id=job_name)


def get_state_counts(batch_job: Any) -> dict[str, int]:
    state = getattr(batch_job, "state", None)
    return {
        "num_requests": int(getattr(state, "num_requests", 0) or 0),
        "num_pending": int(getattr(state, "num_pending", 0) or 0),
        "num_success": int(getattr(state, "num_success", 0) or 0),
        "num_error": int(getattr(state, "num_error", 0) or 0),
        "num_cancelled": int(getattr(state, "num_cancelled", 0) or 0),
    }


def status_batch(args: argparse.Namespace):
    batch_job = get_batch_job(args)
    batch_id = get_object_id(batch_job)
    counts = get_state_counts(batch_job)
    completed = counts["num_success"] + counts["num_error"] + counts["num_cancelled"]
    log_step(f"Batch job: {batch_id}")
    log_step(
        "Current state: "
        f"{completed}/{counts['num_requests']} complete, "
        f"{counts['num_pending']} pending, "
        f"{counts['num_success']} succeeded, "
        f"{counts['num_error']} failed, "
        f"{counts['num_cancelled']} cancelled"
    )
    cost_breakdown = getattr(batch_job, "cost_breakdown", None)
    total_cost_ticks = getattr(cost_breakdown, "total_cost_usd_ticks", None)
    if total_cost_ticks is not None:
        log_step("Total cost so far: $%.4f" % (total_cost_ticks / 1e10))
    return batch_job


def object_to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): object_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_dict(item) for item in value]
    if hasattr(value, "model_dump"):
        return object_to_dict(value.model_dump())
    if hasattr(value, "to_dict"):
        return object_to_dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: object_to_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    usage_dict = object_to_dict(usage)
    if not isinstance(usage_dict, dict):
        usage_dict = {}
    for attr in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_in_usd_ticks",
    ):
        value = getattr(usage, attr, None)
        if value is not None:
            usage_dict.setdefault(attr, value)
    return usage_dict


def write_batch_results(client: Any, batch_id: str, path: Path, page_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pagination_token = None
    with path.open("w", encoding="utf-8") as handle:
        while True:
            page = client.batch.list_batch_results(
                batch_id=batch_id,
                limit=page_size,
                pagination_token=pagination_token,
            )
            for result in getattr(page, "succeeded", []) or []:
                response = getattr(result, "response", None)
                usage = getattr(response, "usage", None)
                row = {
                    "custom_id": str(getattr(result, "batch_request_id", "")),
                    "type": "succeeded",
                    "content": getattr(response, "content", None),
                    "finish_reason": getattr(response, "finish_reason", None),
                    "usage": usage_to_dict(usage),
                    "response": object_to_dict(response),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for result in getattr(page, "failed", []) or []:
                row = {
                    "custom_id": str(getattr(result, "batch_request_id", "")),
                    "type": "failed",
                    "error_message": getattr(result, "error_message", None),
                    "error": object_to_dict(result),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            pagination_token = getattr(page, "pagination_token", None)
            if pagination_token is None:
                break


def extract_text_from_response(result: dict[str, Any]) -> str | None:
    if result.get("type") != "succeeded":
        return None
    content = result.get("content")
    if isinstance(content, str):
        return content
    response = result.get("response") or {}
    response_content = response.get("content")
    return response_content if isinstance(response_content, str) else None


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
    usage = result.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    return input_tokens, output_tokens, total_tokens


def make_result_record(meta: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_response = extract_text_from_response(result)
    pred_label = parse_pred_label(raw_response)
    input_tokens, output_tokens, total_tokens = extract_usage(result)
    error = None
    parse_fail = None

    if result.get("type") != "succeeded":
        error = result.get("error_message") or json.dumps(result.get("error"), ensure_ascii=False)
        parse_fail = {"key": meta["key"], "reason": "request_error", "response": result}
    elif raw_response is None:
        error = "Missing response content"
        parse_fail = {"key": meta["key"], "reason": "missing_response_text", "response": result}
    elif pred_label is None:
        error = "Could not parse label"
        parse_fail = {"key": meta["key"], "reason": "parse_error", "raw_response": raw_response}

    true_label = str(meta["true_label"])
    record = {
        "sample_id": meta["sample_id"],
        "db_name": meta["db_name"],
        "provider": "grok",
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
    batch_id = get_object_id(batch_job)
    counts = get_state_counts(batch_job)
    if counts["num_requests"] == 0:
        raise RuntimeError("Batch job has not loaded any requests yet.")
    if counts["num_pending"] != 0:
        raise RuntimeError(f"Batch job is not complete: {counts['num_pending']} requests still pending")

    redownload = bool(getattr(args, "redownload", False))
    page_size = int(getattr(args, "page_size", 100) or 100)
    if page_size < 1:
        raise ValueError("--page-size must be at least 1")
    if redownload or not result_jsonl.exists():
        with timed_step(f"Download result pages for {batch_id}"):
            write_batch_results(client, batch_id, result_jsonl, page_size)

    ensure_csv_header(paths["result_csv"])
    parse_fails = []
    written = 0
    with result_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            key = str(result.get("custom_id", ""))
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
        counts = get_state_counts(batch_job)
        if counts["num_requests"] > 0 and counts["num_pending"] == 0:
            break
        log_step(f"Sleeping {args.poll_interval_sec}s before next status check")
        time.sleep(args.poll_interval_sec)
    collect_batch(args)
    log_step(f"Done in {format_seconds(time.perf_counter() - started)}")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.model is None:
        args.model = os.getenv("GROK_MODEL", "grok-4.3")

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
