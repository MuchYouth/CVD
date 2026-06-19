"""Inspect trace-aware RAG v2 reranked examples for target samples."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_cvd.evaluation.utils import log_step, timed_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-dataset-root", default="../dataset/juliet_Real_Vul_data.csv")
    parser.add_argument("--target-dataset-csv", default="../dataset/cve_Real_Vul_data.csv")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_train_codebert")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--snippet-chars", type=int, default=220)
    parser.add_argument(
        "--include-full-traces",
        "--include-full-code",
        action="store_true",
        help="Include full query/retrieved processed traces as query_full_trace and retrieved_full_trace.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--output-csv",
        default="results/rag_retrieval_inspection_trace_v2.csv",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.candidate_k < 1:
        raise ValueError("--candidate-k must be at least 1")
    if args.start < 0:
        raise ValueError("--start must be non-negative")

    from llm_cvd.data.juliet_loader import load_juliet_experiment_data
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever
    from llm_cvd.retrieval.rag_retriever_v2 import TraceAwareRagRetriever

    with timed_step("Load RAG train and target datasets"):
        train_records, target_records = load_juliet_experiment_data(
            juliet_root=args.rag_dataset_root,
            real_vul_csv=args.target_dataset_csv,
            cache_dir=args.cache_dir,
            max_train_samples=args.max_train_samples,
            seed=args.seed,
            rebuild_cache=args.rebuild_cache,
        )

    index_name = resolve_index_name(args.index_name, args.rag_dataset_root)
    if args.max_train_samples:
        index_name = f"{index_name}_n{args.max_train_samples}_seed{args.seed}"
    with timed_step(f"Load RAG retriever '{index_name}'"):
        base_retriever = CodeBertFaissRetriever(
            train_records=train_records,
            index_name=index_name,
            index_dir=Path(args.index_dir),
            device=args.device,
        )
        retriever = TraceAwareRagRetriever(
            base_retriever=base_retriever,
            candidate_k=args.candidate_k,
            final_k=args.k,
            train_csv_path=args.rag_dataset_root,
            target_csv_path=args.target_dataset_csv,
        )

    stop = len(target_records) if args.limit is None else min(len(target_records), args.start + args.limit)
    selected = target_records[args.start : stop]
    if not selected:
        raise ValueError("No target records selected.")

    with timed_step(
        f"Retrieve CodeBERT top-{args.candidate_k}, rerank top-{args.k} "
        f"for {len(selected)} target records"
    ):
        retrieved_groups = retriever.retrieve_many_records(
            selected,
            k=args.k,
            batch_size=args.retrieval_batch_size,
        )

    rows = []
    for query_offset, (query, examples) in enumerate(zip(selected, retrieved_groups)):
        query_index = args.start + query_offset
        query_trace = retriever.abstract_query_record(query)
        query_code = str(query["code"])
        for example in examples:
            example_code = str(example.get("code", ""))
            row = {
                "query_sample_id": query.get("sample_id", query_index),
                "query_index": query_index,
                "query_label": query.get("label_text", ""),
                "query_project": query.get("project", ""),
                "query_cwe": query_trace.cwe,
                "query_abstract_trace": query_trace.abstract_trace,
                "retrieved_sample_id": example.get("sample_id", ""),
                "retrieved_index": example.get("retrieved_index", ""),
                "retrieved_label": example.get("label_text", ""),
                "retrieved_project": example.get("project", ""),
                "retrieved_cwe": example.get("cwe_hint", ""),
                "codebert_rank": example.get("codebert_rank", ""),
                "reranked_rank": example.get("reranked_rank", ""),
                "codebert_distance": example.get("codebert_distance", ""),
                "normalized_codebert_distance": example.get("normalized_codebert_distance", ""),
                "trace_score": example.get("trace_score", ""),
                "cwe_score": example.get("cwe_score", ""),
                "source_match": example.get("source_match", ""),
                "sink_match": example.get("sink_match", ""),
                "flow_overlap": example.get("flow_overlap", ""),
                "sanitizer_or_fix_score": example.get("sanitizer_or_fix_score", ""),
                "abstract_trace": example.get("abstract_trace", ""),
                "source_hint": example.get("source_hint", ""),
                "flow_hint": example.get("flow_hint", ""),
                "sink_hint": example.get("sink_hint", ""),
                "root_cause_hint": example.get("root_cause_hint", ""),
                "query_snippet": snippet(query_code, args.snippet_chars),
                "retrieved_snippet": snippet(example_code, args.snippet_chars),
            }
            if args.include_full_traces:
                row.update(
                    {
                        "query_full_trace": query_code,
                        "retrieved_full_trace": example_code,
                    }
                )
            rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log_step(f"Wrote {len(rows)} trace-aware retrieved-example rows to {output_path}")
    print_summary(rows)


def resolve_index_name(index_name: str, rag_dataset_root: str) -> str:
    dataset_path = Path(rag_dataset_root)
    if index_name == "juliet_train_codebert" and dataset_path.is_file():
        return f"{dataset_path.stem}_codebert"
    return index_name


def snippet(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def print_summary(rows: list[dict[str, Any]]) -> None:
    query_count = len({row["query_sample_id"] for row in rows})
    avg_score = sum(float(row["trace_score"]) for row in rows) / len(rows)
    avg_codebert_rank = sum(int(row["codebert_rank"]) for row in rows) / len(rows)
    log_step(
        "Summary: "
        f"queries={query_count}, retrieved_rows={len(rows)}, "
        f"avg_trace_score={avg_score:.4f}, avg_original_codebert_rank={avg_codebert_rank:.2f}"
    )


if __name__ == "__main__":
    main()
