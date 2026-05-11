"""Run API-based zero-shot baseline evaluation and save every provider result immediately."""

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
    log_step,
    parse_model_overrides,
    progress,
    provider_alias_key,
    timed_step,
)
from llm_cvd.prompts.templates import SYSTEM_PROMPT, build_zero_shot_prompt, parse_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="juliet-real")
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
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated API calls per sample/provider.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent API calls.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


def evaluate_client(
    client,
    prompt: str,
    output_sample_id: str,
    db_name: str,
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
        "k": 0,
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

    from llm_cvd.data.juliet_loader import load_real_vul_records
    from llm_cvd.llm.providers import make_client, resolve_provider_model

    provider_aliases = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    model_overrides = parse_model_overrides(args.models)

    log_step("Starting zero-shot baseline API evaluation")
    log_step(f"Provider aliases: {', '.join(provider_aliases)}")
    with timed_step("Load target dataset"):
        test_records = load_real_vul_records(args.target_dataset_csv)
    log_step(f"Loaded {len(test_records)} test records")

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
    run_name = args.run_name or f"{args.db_name}_zeroshot"
    jsonl_path = output_dir / f"{run_name}.jsonl"
    csv_path = output_dir / f"{run_name}.csv"

    completed = load_completed_keys(jsonl_path) if args.resume else set()
    ensure_csv_header(csv_path)
    if completed:
        log_step(f"Resume enabled: loaded {len(completed)} completed result keys")
    log_step(f"Writing results to {jsonl_path} and {csv_path}")

    stop = len(test_records) if args.limit is None else min(len(test_records), args.start + args.limit)
    iterator = range(args.start, stop)
    total_expected = len(iterator) * len(clients) * args.repeats
    skipped = 0
    written = 0
    log_step(
        f"Evaluating samples {args.start}..{stop - 1} "
        f"({len(iterator)} samples x {len(clients)} clients x {args.repeats} repeats = "
        f"{total_expected} attempts)"
    )

    max_workers = max(1, args.max_workers)
    log_step(f"Using up to {max_workers} concurrent API workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sample_id in progress(iterator, desc="Evaluating samples"):
            sample_started = time.perf_counter()
            row = test_records[sample_id]
            output_sample_id = str(row.get("sample_id", sample_id))
            true_label = str(row["label_text"])
            prompt = build_zero_shot_prompt(str(row["code"]))

            futures = []
            for repeat_id in range(args.repeats):
                for client in clients:
                    key = (output_sample_id, client.provider, client.model, repeat_id)
                    if key in completed:
                        skipped += 1
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
                log_step(
                    f"Saved {record['provider']}/{record['model']} sample {output_sample_id} "
                    f"repeat {record['repeat_id']}: {status}, latency={record['latency_sec']:.3f}s"
                )
            log_step(f"Finished sample {output_sample_id} in {format_seconds(time.perf_counter() - sample_started)}")

    log_step(
        f"Done. wrote={written}, skipped={skipped}, "
        f"elapsed={format_seconds(time.perf_counter() - run_started)}"
    )


if __name__ == "__main__":
    main()
