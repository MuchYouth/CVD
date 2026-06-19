"""Trace-aware two-stage RAG retrieval for v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever
from llm_cvd.retrieval.trace_abstractor_v2 import TraceAbstractorV2, TraceFeatures


@dataclass
class TraceAwareRagRetriever:
    """Wrap CodeBERT FAISS retrieval with heuristic trace-aware reranking."""

    base_retriever: CodeBertFaissRetriever
    candidate_k: int = 100
    final_k: int = 6
    train_csv_path: str | Path | None = None
    target_csv_path: str | Path | None = None
    abstractor: TraceAbstractorV2 | None = None
    train_abstractor: TraceAbstractorV2 | None = None
    target_abstractor: TraceAbstractorV2 | None = None
    train_trace_cache: dict[int, TraceFeatures] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.final_k < 1:
            raise ValueError("final_k must be at least 1")
        if self.abstractor is not None:
            self.train_abstractor = self.train_abstractor or self.abstractor
            self.target_abstractor = self.target_abstractor or self.abstractor
        self.train_abstractor = self.train_abstractor or TraceAbstractorV2(
            csv_paths=[self.train_csv_path] if self.train_csv_path is not None else []
        )
        self.target_abstractor = self.target_abstractor or TraceAbstractorV2(
            csv_paths=[self.target_csv_path] if self.target_csv_path is not None else []
        )

    @classmethod
    def from_train_records(
        cls,
        train_records: list[dict[str, Any]],
        index_name: str,
        index_dir: Path,
        candidate_k: int = 100,
        final_k: int = 6,
        train_csv_path: str | Path | None = None,
        target_csv_path: str | Path | None = None,
        rebuild_index: bool = False,
        device: str | None = None,
    ) -> "TraceAwareRagRetriever":
        base = CodeBertFaissRetriever(
            train_records=train_records,
            index_name=index_name,
            index_dir=index_dir,
            rebuild_index=rebuild_index,
            device=device,
        )
        return cls(
            base_retriever=base,
            candidate_k=candidate_k,
            final_k=final_k,
            train_csv_path=train_csv_path,
            target_csv_path=target_csv_path,
        )

    def retrieve_record(self, record: dict[str, Any], k: int | None = None) -> list[dict[str, Any]]:
        embedding = np.array([self.base_retriever.get_embedding(str(record["code"]))], dtype=np.float32)
        return self._search_and_rerank(record, embedding[0], k or self.final_k)

    def retrieve_many_records(
        self,
        records: list[dict[str, Any]],
        k: int | None = None,
        batch_size: int = 16,
        desc: str = "Embedding target records",
    ) -> list[list[dict[str, Any]]]:
        if not records:
            return []
        embeddings = self.base_retriever.get_embeddings(
            [str(record["code"]) for record in records],
            batch_size=batch_size,
            desc=desc,
        )
        final_k = k or self.final_k
        return [
            self._search_and_rerank(record, embedding, final_k)
            for record, embedding in zip(records, embeddings)
        ]

    def retrieve_many(
        self,
        codes: list[str],
        k: int = 6,
        batch_size: int = 16,
        desc: str = "Embedding target records",
    ) -> list[list[dict[str, Any]]]:
        records = [{"code": code, "sample_id": str(index)} for index, code in enumerate(codes)]
        return self.retrieve_many_records(records, k=k, batch_size=batch_size, desc=desc)

    def abstract_query_record(self, record: dict[str, Any]) -> TraceFeatures:
        return self.target_abstractor.abstract_record(record)

    def _search_and_rerank(
        self,
        query_record: dict[str, Any],
        query_embedding: np.ndarray,
        final_k: int,
    ) -> list[dict[str, Any]]:
        search_k = min(self.candidate_k, len(self.base_retriever.metadata))
        distances, indices = self.base_retriever.index.search(
            np.array([query_embedding], dtype=np.float32),
            search_k,
        )
        candidates = []
        valid_distances = [float(distance) for distance, idx in zip(distances[0], indices[0]) if int(idx) >= 0]
        query_trace = self.abstract_query_record(query_record)
        for codebert_rank, (distance, index_value) in enumerate(zip(distances[0], indices[0]), start=1):
            metadata_index = int(index_value)
            if metadata_index < 0 or metadata_index >= len(self.base_retriever.metadata):
                continue
            example = self.base_retriever.metadata[metadata_index]
            candidate_trace = self._train_trace(metadata_index, example)
            score_parts = trace_score(query_trace, candidate_trace, float(distance), valid_distances)
            enriched = dict(example)
            enriched.update(
                {
                    "retrieved_index": metadata_index,
                    "codebert_rank": codebert_rank,
                    "codebert_distance": float(distance),
                    "trace_score": score_parts["final_score"],
                    "source_match": score_parts["source_score"],
                    "sink_match": score_parts["sink_score"],
                    "flow_overlap": score_parts["flow_score"],
                    "cwe_score": score_parts["cwe_score"],
                    "sanitizer_or_fix_score": score_parts["sanitizer_or_fix_score"],
                    "normalized_codebert_distance": score_parts["normalized_codebert_distance"],
                    "abstract_trace": candidate_trace.abstract_trace,
                    "source_hint": candidate_trace.source_hint,
                    "flow_hint": candidate_trace.flow_hint,
                    "sink_hint": candidate_trace.sink_hint,
                    "root_cause_hint": candidate_trace.root_cause_hint,
                    "cwe_hint": candidate_trace.cwe,
                    "target_abstract_trace": query_trace.abstract_trace,
                    "target_source_hint": query_trace.source_hint,
                    "target_flow_hint": query_trace.flow_hint,
                    "target_sink_hint": query_trace.sink_hint,
                    "trace_features_v2": candidate_trace.to_dict(),
                }
            )
            candidates.append(enriched)

        candidates.sort(
            key=lambda row: (
                -float(row["trace_score"]),
                float(row["codebert_distance"]),
                int(row["codebert_rank"]),
            )
        )
        for reranked_rank, row in enumerate(candidates, start=1):
            row["reranked_rank"] = reranked_rank
        return candidates[:final_k]

    def _train_trace(self, metadata_index: int, record: dict[str, Any]) -> TraceFeatures:
        if metadata_index not in self.train_trace_cache:
            self.train_trace_cache[metadata_index] = self.train_abstractor.abstract_record(record)
        return self.train_trace_cache[metadata_index]


def trace_score(
    query: TraceFeatures,
    candidate: TraceFeatures,
    codebert_distance: float,
    candidate_distances: list[float],
) -> dict[str, float]:
    source_score = grouped_overlap(query.source_apis, candidate.source_apis, query.source_kinds, candidate.source_kinds)
    sink_score = grouped_overlap(query.sink_apis, candidate.sink_apis, query.sink_kinds, candidate.sink_kinds)
    cwe_score = cwe_overlap(query.cwe, candidate.cwe)
    flow_score = grouped_overlap(
        query.flow_apis | query.controls,
        candidate.flow_apis | candidate.controls,
        query.flow_kinds,
        candidate.flow_kinds,
    )
    sanitizer_or_fix_score = grouped_overlap(
        query.sanitizer_apis,
        candidate.sanitizer_apis,
        query.sanitizer_kinds,
        candidate.sanitizer_kinds,
    )
    normalized_distance = normalize_distance(codebert_distance, candidate_distances)
    final_score = (
        0.35 * sink_score
        + 0.25 * source_score
        + 0.20 * cwe_score
        + 0.15 * flow_score
        + 0.05 * sanitizer_or_fix_score
        - 0.10 * normalized_distance
    )
    return {
        "final_score": round(final_score, 6),
        "sink_score": round(sink_score, 6),
        "source_score": round(source_score, 6),
        "cwe_score": round(cwe_score, 6),
        "flow_score": round(flow_score, 6),
        "sanitizer_or_fix_score": round(sanitizer_or_fix_score, 6),
        "normalized_codebert_distance": round(normalized_distance, 6),
    }


def grouped_overlap(
    left_exact: set[str],
    right_exact: set[str],
    left_groups: set[str],
    right_groups: set[str],
) -> float:
    exact = jaccard(left_exact, right_exact)
    group = jaccard(left_groups, right_groups)
    if left_exact & right_exact:
        return 1.0
    if left_groups & right_groups:
        return max(0.6, group)
    return max(exact, group)


def cwe_overlap(left: str, right: str) -> float:
    left_values = cwe_values(left)
    right_values = cwe_values(right)
    if not left_values or not right_values:
        return 0.0
    if left_values & right_values:
        return 1.0
    left_families = {value.split("-", 1)[-1][:2] for value in left_values}
    right_families = {value.split("-", 1)[-1][:2] for value in right_values}
    return 0.5 if left_families & right_families else 0.0


def cwe_values(value: str) -> set[str]:
    return {part.strip().upper() for part in str(value or "").split(";") if part.strip()}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def normalize_distance(distance: float, distances: list[float]) -> float:
    if not distances:
        return 0.0
    min_distance = min(distances)
    max_distance = max(distances)
    if max_distance <= min_distance:
        return 0.0
    return (distance - min_distance) / (max_distance - min_distance)
