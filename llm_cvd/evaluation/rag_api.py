"""Run API-based few-shot RAG evaluation and save every provider result immediately."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_cvd.evaluation.utils import (
    append_record,
    ensure_csv_header,
    format_seconds,
    load_completed_keys,
    load_env_file,
    load_result_key_sets,
    log_step,
    parse_model_overrides,
    provider_alias_key,
    timed_step,
)
from llm_cvd.prompts.templates import SYSTEM_PROMPT, build_few_shot_prompt, parse_label


class NoOpProgressBar:
    def update(self, count: int = 1) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress_bar(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="call", dynamic_ncols=True)
    except ModuleNotFoundError:
        return NoOpProgressBar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="juliet-real")
    parser.add_argument(
        "--rag-dataset-root",
        "--juliet-root",
        dest="rag_dataset_root",
        default="../../juliet-playground/juliet-test-suite-v1.3",
        help="Root directory for raw Juliet testcases, or a Real_Vul-style CSV for RAG training.",
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
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated API calls per sample/provider.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent API calls.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-errors-from",
        default=None,
        help=(
            "Optional CSV/JSONL result file. When set, only calls that errored in this "
            "file and do not already have a successful row in this file are queued."
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_train_codebert")
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def evaluate_client(
    client,
    prompt: str,
    output_sample_id: str,
    db_name: str,
    k: int,
    true_label: str,
    repeat_id: int,
    max_output_tokens: int,
) -> tuple[tuple[str, str, str, int], dict[str, object], str]:
    result = client.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=max_output_tokens,
    )
    pred_label = parse_label(result.text)
    key = (output_sample_id, result.provider, result.model, repeat_id)
    record = {
        "sample_id": output_sample_id,
        "db_name": db_name,
        "provider": result.provider,
        "model": result.model,
        "k": k,
        "repeat_id": repeat_id,
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
    status = "error" if result.error else f"pred={pred_label}"
    return key, record, status


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    load_env_file(args.env_file)

    from llm_cvd.data.juliet_loader import load_juliet_experiment_data
    from llm_cvd.llm.providers import make_client, resolve_provider_model
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever

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

    retry_keys = None
    retry_sample_ids = None
    if args.retry_errors_from:
        retry_path = Path(args.retry_errors_from)
        with timed_step(f"Load retry keys from {retry_path}"):
            source_successful, source_errored = load_result_key_sets(retry_path)
        retry_keys = source_errored - source_successful
        retry_sample_ids = {key[0] for key in retry_keys}
        log_step(
            f"Retry-errors mode: {len(retry_keys)} failed calls across "
            f"{len(retry_sample_ids)} samples will be eligible for retry"
        )

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
    sample_ids = list(iterator)
    if retry_sample_ids is not None:
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if str(test_records[sample_id].get("sample_id", sample_id)) in retry_sample_ids
        ]
    total_expected = len(sample_ids) * len(clients) * args.repeats
    skipped = 0
    written = 0
    log_step(
        f"Evaluating samples {args.start}..{stop - 1} "
        f"({len(sample_ids)} selected samples x {len(clients)} clients x {args.repeats} repeats = "
        f"{total_expected} attempts)"
    )

    selected_rows = [test_records[sample_id] for sample_id in sample_ids]
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

    max_workers = max(1, args.max_workers)
    log_step(f"Using up to {max_workers} concurrent API workers")
    api_progress = make_progress_bar(total_expected, "API calls")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            for sample_id, row, prompt in zip(sample_ids, selected_rows, prompts):
                sample_started = time.perf_counter()
                output_sample_id = str(row.get("sample_id", sample_id))
                true_label = str(row["label_text"])

                futures = []
                for repeat_id in range(args.repeats):
                    for client in clients:
                        key = (output_sample_id, client.provider, client.model, repeat_id)
                        if retry_keys is not None and key not in retry_keys:
                            skipped += 1
                            api_progress.update(1)
                            api_progress.set_postfix(written=written, skipped=skipped)
                            continue
                        if key in completed:
                            skipped += 1
                            api_progress.update(1)
                            api_progress.set_postfix(written=written, skipped=skipped)
                            continue

                        log_step(
                            f"Queueing {client.provider}/{client.model} for sample {output_sample_id} "
                            f"(repeat {repeat_id + 1}/{args.repeats})"
                        )
                        futures.append(
                            executor.submit(
                                evaluate_client,
                                client,
                                prompt,
                                output_sample_id,
                                args.db_name,
                                args.k,
                                true_label,
                                repeat_id,
                                args.max_output_tokens,
                            )
                        )

                for future in as_completed(futures):
                    key, record, status = future.result()
                    append_record(jsonl_path, csv_path, record)
                    completed.add(key)
                    written += 1
                    api_progress.update(1)
                    api_progress.set_postfix(written=written, skipped=skipped)
                    log_step(
                        f"Saved {record['provider']}/{record['model']} sample {output_sample_id} "
                        f"repeat {record['repeat_id']}: {status}, latency={record['latency_sec']:.3f}s"
                    )
                log_step(
                    f"Finished sample {output_sample_id} in "
                    f"{format_seconds(time.perf_counter() - sample_started)}"
                )
        finally:
            api_progress.close()

    log_step(
        f"Done. wrote={written}, skipped={skipped}, "
        f"elapsed={format_seconds(time.perf_counter() - run_started)}"
    )


if __name__ == "__main__":
    main()
