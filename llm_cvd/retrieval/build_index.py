"""Build a CodeBERT FAISS index for RAG experiments without calling APIs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_cvd.evaluation.utils import format_seconds, log_step, timed_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag-dataset-root",
        "--juliet-root",
        dest="rag_dataset_root",
        default="../../juliet-playground/juliet-test-suite-v1.3",
        help="Root directory for raw Juliet testcases, or a Real_Vul-style CSV for RAG training.",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_train_codebert")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--index-batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, for example 'cuda' or 'cpu'. Defaults to auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.index_batch_size < 1:
        raise ValueError("--index-batch-size must be at least 1")

    from llm_cvd.data.juliet_loader import load_juliet_train_records
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever

    log_step("Starting RAG index build")
    with timed_step("Load RAG training dataset"):
        train_records = load_juliet_train_records(
            juliet_root=args.rag_dataset_root,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
        )

    if args.max_train_samples and len(train_records) > args.max_train_samples:
        import random

        log_step(
            f"Sampling {args.max_train_samples} train records from {len(train_records)} "
            f"with seed={args.seed}"
        )
        rng = random.Random(args.seed)
        train_records = train_records[:]
        rng.shuffle(train_records)
        train_records = train_records[: args.max_train_samples]

    index_name = args.index_name
    rag_dataset_path = Path(args.rag_dataset_root)
    if index_name == "juliet_train_codebert" and rag_dataset_path.is_file():
        index_name = f"{rag_dataset_path.stem}_codebert"
    if args.max_train_samples:
        index_name = f"{index_name}_n{args.max_train_samples}_seed{args.seed}"

    index_dir = Path(args.index_dir)
    log_step(f"Training records: {len(train_records)}")
    log_step(f"Index name: {index_name}")
    log_step(f"Index directory: {index_dir}")

    with timed_step(f"Build or load FAISS index '{index_name}'"):
        retriever = CodeBertFaissRetriever(
            train_records=train_records,
            index_name=index_name,
            index_dir=index_dir,
            device=args.device,
            rebuild_index=args.rebuild_index,
            index_batch_size=args.index_batch_size,
        )

    log_step(f"Index vectors: {retriever.index.ntotal}")
    log_step(f"Metadata records: {len(retriever.metadata)}")
    log_step(f"Done in {format_seconds(time.perf_counter() - run_started)}")


if __name__ == "__main__":
    main()
