"""CodeBERT embedding and FAISS retrieval for Juliet RAG experiments."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PACKAGE_DIR / "indexes"


@dataclass
class CodeBertFaissRetriever:
    """Retrieve similar Juliet train examples using CodeBERT and FAISS."""

    train_records: list[dict[str, Any]]
    index_name: str = "juliet_train_codebert"
    index_dir: Path = DEFAULT_INDEX_DIR
    embedding_model_name: str = "microsoft/codebert-base"
    device: str | None = None
    rebuild_index: bool = False
    index_batch_size: int = 16
    embedding_max_length: int = 512
    embedding_truncation_side: str = "left"

    def __post_init__(self) -> None:
        if self.embedding_truncation_side not in {"left", "right"}:
            raise ValueError("embedding_truncation_side must be 'left' or 'right'")
        if self.embedding_max_length < 1:
            raise ValueError("embedding_max_length must be at least 1")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log_retriever(f"Using embedding device: {self.device}")
        started = time.perf_counter()
        log_retriever(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_tokenizer.truncation_side = self.embedding_truncation_side
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        self.embedding_model.eval()
        kept_side = "tail" if self.embedding_truncation_side == "left" else "head"
        log_retriever(
            f"Embedding truncation: keep {kept_side} "
            f"{self.embedding_max_length} tokens"
        )
        log_retriever(f"Loaded embedding model in {format_seconds(time.perf_counter() - started)}")
        self.index, self.metadata = self._load_or_build_index()

    def retrieve(self, code: str, k: int = 6) -> list[dict[str, Any]]:
        embedding = np.array([self.get_embedding(code)], dtype=np.float32)
        _, indices = self.index.search(embedding, k)
        return self._metadata_from_indices(indices[0])

    def retrieve_many(
        self,
        codes: list[str],
        k: int = 6,
        batch_size: int = 16,
        desc: str = "Embedding target records",
    ) -> list[list[dict[str, Any]]]:
        if not codes:
            return []
        embeddings = self.get_embeddings(codes, batch_size=batch_size, desc=desc)
        _, indices = self.index.search(embeddings, k)
        return [self._metadata_from_indices(row_indices) for row_indices in indices]

    def _metadata_from_indices(self, indices: np.ndarray) -> list[dict[str, Any]]:
        examples = []
        for idx in indices:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(self.metadata):
                continue
            examples.append(self.metadata[idx_int])
        return examples

    def get_embedding(self, code_snippet: str) -> np.ndarray:
        return self.get_embeddings([code_snippet], batch_size=1)[0]

    def get_embeddings(
        self,
        code_snippets: list[str],
        batch_size: int = 16,
        desc: str | None = None,
    ) -> np.ndarray:
        embeddings = []
        batch_size = max(1, batch_size)
        batch_starts = range(0, len(code_snippets), batch_size)
        if desc:
            batch_starts = progress(batch_starts, desc=desc)
        for start in batch_starts:
            batch = code_snippets[start : start + batch_size]
            embeddings.append(self._embed_batch(batch))
        return np.vstack(embeddings).astype(np.float32)

    def _embed_batch(self, code_snippets: list[str]) -> np.ndarray:
        inputs = self.embedding_tokenizer(
            code_snippets,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.embedding_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    def _load_or_build_index(self) -> tuple[faiss.Index, list[dict[str, Any]]]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"{self.index_name}.index"
        metadata_path = self.index_dir / f"{self.index_name}_metadata.pkl"
        config_path = self.index_dir / f"{self.index_name}_embedding_config.json"

        if index_path.exists() and metadata_path.exists() and not self.rebuild_index:
            if self._index_config_matches(config_path):
                return self._read_index(index_path, metadata_path)
            log_retriever(
                "Existing FAISS index embedding config is missing or different; "
                "rebuilding index"
            )

        if not self.train_records:
            raise ValueError("No Juliet train records were loaded.")

        started = time.perf_counter()
        log_retriever(f"Building embeddings for {len(self.train_records)} train records")
        train_codes = [str(record["code"]) for record in self.train_records]
        embeddings = self.get_embeddings(
            train_codes,
            batch_size=self.index_batch_size,
            desc="Embedding Juliet records",
        )
        log_retriever(f"Built embeddings in {format_seconds(time.perf_counter() - started)}")
        started = time.perf_counter()
        log_retriever(f"Building FAISS L2 index with dimension {embeddings.shape[1]}")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        with metadata_path.open("wb") as handle:
            pickle.dump(self.train_records, handle)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(self._embedding_config(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        log_retriever(
            f"Saved FAISS index and metadata in {format_seconds(time.perf_counter() - started)}"
        )
        return index, self.train_records

    def _read_index(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> tuple[faiss.Index, list[dict[str, Any]]]:
        started = time.perf_counter()
        log_retriever(f"Reading FAISS index: {index_path}")
        index = faiss.read_index(str(index_path))
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        log_retriever(
            f"Loaded FAISS index with {len(metadata)} records "
            f"in {format_seconds(time.perf_counter() - started)}"
        )
        return index, metadata

    def _index_config_matches(self, config_path: Path) -> bool:
        if not config_path.exists():
            return False
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle) == self._embedding_config()
        except (OSError, json.JSONDecodeError):
            return False

    def _embedding_config(self) -> dict[str, Any]:
        return {
            "embedding_model_name": self.embedding_model_name,
            "embedding_max_length": self.embedding_max_length,
            "embedding_truncation_side": self.embedding_truncation_side,
        }


def log_retriever(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [rag] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


def progress(items, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ModuleNotFoundError:
        return items
