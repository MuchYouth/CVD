"""Inspect the top-k RAG examples retrieved for target vulnerability samples."""

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
    parser.add_argument(
        "--rag-dataset-root",
        default="../dataset/juliet_Real_Vul_data.csv",
        help="Training dataset used to build the RAG index.",
    )
    parser.add_argument(
        "--target-dataset-csv",
        default="../dataset/cve_Real_Vul_data.csv",
        help="Target CVE/Real_Vul-style CSV to query against the RAG index.",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_train_codebert")
    parser.add_argument("--k", type=int, default=6)
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
    parser.add_argument(
        "--output-csv",
        default="results/rag_retrieval_inspection.csv",
        help="CSV path for the flattened query/retrieved-example report.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, for example 'cuda' or 'cpu'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.retrieval_batch_size < 1:
        raise ValueError("--retrieval-batch-size must be at least 1")
    if args.start < 0:
        raise ValueError("--start must be non-negative")

    from llm_cvd.data.juliet_loader import load_juliet_experiment_data
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever

    with timed_step("Load RAG train and target datasets"):
        train_records, target_records = load_juliet_experiment_data(
            juliet_root=args.rag_dataset_root,
            real_vul_csv=args.target_dataset_csv,
            cache_dir=args.cache_dir,
        )

    index_name = resolve_index_name(args.index_name, args.rag_dataset_root)
    with timed_step(f"Load RAG retriever '{index_name}'"):
        retriever = CodeBertFaissRetriever(
            train_records=train_records,
            index_name=index_name,
            index_dir=Path(args.index_dir),
            device=args.device,
        )

    stop = len(target_records) if args.limit is None else min(len(target_records), args.start + args.limit)
    selected = target_records[args.start : stop]
    if not selected:
        raise ValueError("No target records selected.")

    with timed_step(f"Retrieve top-{args.k} examples for {len(selected)} target records"):
        query_embeddings = retriever.get_embeddings(
            [str(row["code"]) for row in selected],
            batch_size=args.retrieval_batch_size,
            desc="Embedding target records",
        )
        distances, indices = retriever.index.search(query_embeddings, args.k)

    rows = []
    tokenizer = retriever.embedding_tokenizer
    embedding_window = (
        f"{'tail' if retriever.embedding_truncation_side == 'left' else 'head'}_"
        f"{retriever.embedding_max_length}"
    )
    for query_offset, query in enumerate(selected):
        query_code = str(query["code"])
        query_tokens = count_tokens(tokenizer, query_code)
        query_cve = normalize_cve(query.get("cve", ""))
        for rank, (distance, retrieved_index) in enumerate(
            zip(distances[query_offset], indices[query_offset]),
            start=1,
        ):
            example = retriever.metadata[int(retrieved_index)]
            example_code = str(example["code"])
            retrieved_cve = example.get("cve", "")
            retrieved_cve_normalized = normalize_cve(retrieved_cve)
            row = {
                "query_sample_id": query.get("sample_id", args.start + query_offset),
                "query_index": args.start + query_offset,
                "query_label": query.get("label_text", ""),
                "query_project": query.get("project", ""),
                "query_cve": query_cve,
                "query_code_chars": len(query_code),
                "query_code_lines": line_count(query_code),
                "query_codebert_tokens": query_tokens,
                "query_truncated_for_embedding": query_tokens > retriever.embedding_max_length,
                "embedding_token_window": embedding_window,
                "rank": rank,
                "faiss_l2_distance": float(distance),
                "retrieved_index": int(retrieved_index),
                "retrieved_sample_id": example.get("sample_id", ""),
                "retrieved_label": example.get("label_text", ""),
                "label_matches_query": example.get("label_text", "") == query.get("label_text", ""),
                "retrieved_project": example.get("project", ""),
                "retrieved_cve": retrieved_cve,
                "retrieved_cve_normalized": retrieved_cve_normalized,
                "same_cve": cve_matches(query_cve, retrieved_cve_normalized),
                "retrieved_dataset_type": example.get("dataset_type", ""),
                "retrieved_source": example.get("source", ""),
                "retrieved_code_chars": len(example_code),
                "retrieved_code_lines": line_count(example_code),
                "retrieved_codebert_tokens": count_tokens(tokenizer, example_code),
                "lexical_jaccard": lexical_jaccard(query_code, example_code),
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

    log_step(f"Wrote {len(rows)} retrieved-example rows to {output_path}")
    print_summary(rows)


def resolve_index_name(index_name: str, rag_dataset_root: str) -> str:
    dataset_path = Path(rag_dataset_root)
    if index_name == "juliet_train_codebert" and dataset_path.is_file():
        return f"{dataset_path.stem}_codebert"
    return index_name


def count_tokens(tokenizer: Any, code: str) -> int:
    return len(tokenizer.encode(code, add_special_tokens=True, truncation=False))


def line_count(text: str) -> int:
    return text.count("\n") + 1 if text else 0


def snippet(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def normalize_cve(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "UNKNOWN"


def cve_values(value: str) -> set[str]:
    if not value or value == "UNKNOWN":
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def cve_matches(query_cve: str, retrieved_cve: str) -> bool:
    query_values = cve_values(query_cve)
    retrieved_values = cve_values(retrieved_cve)
    return bool(query_values & retrieved_values)


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def lexical_tokens(code: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
        if len(token) > 1
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    query_count = len({row["query_sample_id"] for row in rows})
    label_matches = sum(1 for row in rows if row["label_matches_query"])
    truncated = len({row["query_sample_id"] for row in rows if row["query_truncated_for_embedding"]})
    avg_distance = sum(float(row["faiss_l2_distance"]) for row in rows) / len(rows)
    avg_jaccard = sum(float(row["lexical_jaccard"]) for row in rows) / len(rows)
    log_step(
        "Summary: "
        f"queries={query_count}, retrieved_rows={len(rows)}, "
        f"label_match_rate={label_matches / len(rows):.3f}, "
        f"truncated_queries={truncated}, "
        f"avg_l2_distance={avg_distance:.4f}, "
        f"avg_lexical_jaccard={avg_jaccard:.4f}"
    )


if __name__ == "__main__":
    main()
