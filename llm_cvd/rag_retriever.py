"""CodeBERT embedding and FAISS retrieval for Juliet RAG experiments."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = THIS_DIR / "indexes"


@dataclass
class CodeBertFaissRetriever:
    """Retrieve similar Juliet train examples using CodeBERT and FAISS."""

    train_records: list[dict[str, Any]]
    index_name: str = "juliet_train_codebert"
    index_dir: Path = DEFAULT_INDEX_DIR
    embedding_model_name: str = "microsoft/codebert-base"
    device: str | None = None
    rebuild_index: bool = False

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log_retriever(f"Using embedding device: {self.device}")
        started = time.perf_counter()
        log_retriever(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        self.embedding_model.eval()
        log_retriever(f"Loaded embedding model in {format_seconds(time.perf_counter() - started)}")
        self.index, self.metadata = self._load_or_build_index()

    def retrieve(self, code: str, k: int = 6) -> list[dict[str, Any]]:
        embedding = np.array([self.get_embedding(code)], dtype=np.float32)
        _, indices = self.index.search(embedding, k)
        examples = []
        for idx in indices[0]:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(self.metadata):
                continue
            examples.append(self.metadata[idx_int])
        return examples

    def get_embedding(self, code_snippet: str) -> np.ndarray:
        inputs = self.embedding_tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

    def _load_or_build_index(self) -> tuple[faiss.Index, list[dict[str, Any]]]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"{self.index_name}.index"
        metadata_path = self.index_dir / f"{self.index_name}_metadata.pkl"

        if index_path.exists() and metadata_path.exists() and not self.rebuild_index:
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

        if not self.train_records:
            raise ValueError("No Juliet train records were loaded.")

        started = time.perf_counter()
        log_retriever(f"Building embeddings for {len(self.train_records)} train records")
        embeddings = np.array(
            [
                self.get_embedding(record["code"])
                for record in progress(self.train_records, desc="Embedding Juliet records")
            ],
            dtype=np.float32,
        )
        log_retriever(f"Built embeddings in {format_seconds(time.perf_counter() - started)}")
        started = time.perf_counter()
        log_retriever(f"Building FAISS L2 index with dimension {embeddings.shape[1]}")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        with metadata_path.open("wb") as handle:
            pickle.dump(self.train_records, handle)
        log_retriever(
            f"Saved FAISS index and metadata in {format_seconds(time.perf_counter() - started)}"
        )
        return index, self.train_records


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
