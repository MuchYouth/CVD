"""Create a t-SNE view of CVE query embeddings and retrieved Juliet examples."""

from __future__ import annotations

import argparse
import csv
import html
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-csv", default="results/rag_retrieval_inspection_cve_k6.csv")
    parser.add_argument("--target-dataset-csv", default="../dataset/cve_Real_Vul_data.csv")
    parser.add_argument("--rag-dataset-root", default="../dataset/juliet_Real_Vul_data.csv")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--index-name", default="juliet_Real_Vul_data_codebert")
    parser.add_argument("--embedding-model-name", default="microsoft/codebert-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--background-sample", type=int, default=250)
    parser.add_argument("--corpus-label", default="Juliet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output-csv", default="results/rag_tsne_cve_k6_points.csv")
    parser.add_argument("--output-html", default="results/rag_tsne_cve_k6.html")
    parser.add_argument("--output-png", default="results/rag_tsne_cve_k6.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cvd")
    import faiss
    from sklearn.manifold import TSNE

    from llm_cvd.data.juliet_loader import load_juliet_csv_train_records, load_real_vul_records
    from llm_cvd.retrieval.rag_retriever import CodeBertFaissRetriever

    inspection_rows = list(csv.DictReader(open(args.inspection_csv, newline="", encoding="utf-8")))
    retrieved_ids = {row["retrieved_sample_id"] for row in inspection_rows}
    retrieved_indices = {int(row["retrieved_index"]) for row in inspection_rows}

    train_records = load_juliet_csv_train_records(Path(args.rag_dataset_root))
    query_records = load_real_vul_records(args.target_dataset_csv)

    index_path = Path(args.index_dir) / f"{args.index_name}.index"
    metadata_path = Path(args.index_dir) / f"{args.index_name}_metadata.pkl"
    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    rng = np.random.default_rng(args.seed)
    background_candidates = np.array(
        [idx for idx in range(index.ntotal) if idx not in retrieved_indices],
        dtype=np.int64,
    )
    sample_size = min(args.background_sample, len(background_candidates))
    background_indices = set(rng.choice(background_candidates, size=sample_size, replace=False).tolist())
    selected_indices = sorted(background_indices | retrieved_indices)

    selected_train_vectors = np.vstack([index.reconstruct(idx) for idx in selected_indices])

    retriever = CodeBertFaissRetriever(
        train_records=train_records,
        index_name=args.index_name,
        index_dir=Path(args.index_dir),
        embedding_model_name=args.embedding_model_name,
        device=args.device,
    )
    query_vectors = retriever.get_embeddings(
        [str(row["code"]) for row in query_records],
        batch_size=16,
        desc="Embedding CVE query records",
    )

    vectors = np.vstack([selected_train_vectors, query_vectors])
    labels = build_point_rows(
        selected_indices=selected_indices,
        metadata=metadata,
        retrieved_ids=retrieved_ids,
        query_records=query_records,
    )

    perplexity = min(args.perplexity, max(5.0, (len(vectors) - 1) / 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(vectors)

    for row, coord in zip(labels, coords):
        row["x"] = f"{coord[0]:.6f}"
        row["y"] = f"{coord[1]:.6f}"

    write_points_csv(Path(args.output_csv), labels)
    write_points_html(Path(args.output_html), labels, args.corpus_label)
    write_points_png(Path(args.output_png), labels, args.corpus_label)
    print(f"Wrote t-SNE point CSV: {args.output_csv}")
    print(f"Wrote t-SNE HTML: {args.output_html}")
    print(f"Wrote t-SNE PNG: {args.output_png}")
    print(f"Points: {len(labels)} ({len(query_records)} queries, {len(retrieved_ids)} unique retrieved examples)")


def build_point_rows(
    selected_indices: list[int],
    metadata: list[dict],
    retrieved_ids: set[str],
    query_records: list[dict],
) -> list[dict[str, str]]:
    rows = []
    for idx in selected_indices:
        record = metadata[idx]
        sample_id = str(record.get("sample_id", ""))
        rows.append(
            {
                "kind": "retrieved_juliet" if sample_id in retrieved_ids else "background_juliet",
                "sample_id": sample_id,
                "project": str(record.get("project", "")),
                "cwe_or_hint": cwe_or_hint(sample_id, str(record.get("code", ""))),
                "label": str(record.get("label_text", "")),
                "source_index": str(idx),
                "x": "",
                "y": "",
            }
        )
    for record in query_records:
        rows.append(
            {
                "kind": "query_cve",
                "sample_id": str(record.get("sample_id", "")),
                "project": str(record.get("project", "")),
                "cwe_or_hint": "",
                "label": str(record.get("label_text", "")),
                "source_index": "",
                "x": "",
                "y": "",
            }
        )
    return rows


def cwe_or_hint(sample_id: str, code: str) -> str:
    lowered = code.lower()
    if "rand32" in lowered or "urand31" in lowered:
        return "random integer/count"
    if "command_injection" in lowered or "system(" in lowered:
        return "OS command injection"
    if "fscanf" in lowered and any(token in lowered for token in ["strncpy", "memcpy", "memmove"]):
        return "buffer copy/input length"
    if "fwrite" in lowered:
        return "file/count"
    return sample_id


def write_points_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_points_html(path: Path, rows: list[dict[str, str]], corpus_label: str) -> None:
    width = 1000
    height = 720
    pad = 48
    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def sx(value: str) -> float:
        if max_x == min_x:
            return width / 2
        return pad + (float(value) - min_x) / (max_x - min_x) * (width - 2 * pad)

    def sy(value: str) -> float:
        if max_y == min_y:
            return height / 2
        return height - pad - (float(value) - min_y) / (max_y - min_y) * (height - 2 * pad)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG t-SNE CVE K6</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; color: #1f2933; }}
svg {{ border: 1px solid #d0d7de; background: #ffffff; }}
.legend {{ display: flex; gap: 18px; margin: 12px 0; align-items: center; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
.bg {{ fill: #009e73; opacity: 0.55; }}
.retrieved {{ fill: #f59e0b; stroke: #7c2d12; stroke-width: 1; }}
.query {{ fill: #2563eb; stroke: #172554; stroke-width: 1.5; }}
text {{ font-size: 11px; }}
</style>
</head>
<body>
<h1>t-SNE of CVE query embeddings and {html.escape(corpus_label)} retrieved examples</h1>
<div class="legend">
<span><span class="dot" style="background:#009e73"></span>Background {html.escape(corpus_label)} sample</span>
<span><span class="dot" style="background:#f59e0b"></span>Unique retrieved {html.escape(corpus_label)} example</span>
<span><span class="dot" style="background:#2563eb"></span>CVE query</span>
</div>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
"""
        )
        for row in rows:
            kind = row["kind"]
            css = "query" if kind == "query_cve" else "retrieved" if kind == "retrieved_juliet" else "bg"
            radius = 7 if kind == "query_cve" else 5 if kind == "retrieved_juliet" else 3
            title = " | ".join(
                value
                for value in [kind, row["sample_id"], row["project"], row["cwe_or_hint"], row["label"]]
                if value
            )
            handle.write(
                f'<circle class="{css}" cx="{sx(row["x"]):.2f}" cy="{sy(row["y"]):.2f}" r="{radius}">'
                f"<title>{html.escape(title)}</title></circle>\n"
            )
            if kind == "query_cve":
                handle.write(
                    f'<text x="{sx(row["x"]) + 8:.2f}" y="{sy(row["y"]) + 4:.2f}">'
                    f"Q{html.escape(row['sample_id'])}</text>\n"
                )
        handle.write("</svg>\n</body>\n</html>\n")


def write_points_png(path: Path, rows: list[dict[str, str]], corpus_label: str) -> None:
    import matplotlib.pyplot as plt

    groups = {
        "background_juliet": {
            "label": f"Background {corpus_label} sample",
            "color": "#009e73",
            "size": 18,
            "alpha": 0.55,
            "marker": "o",
        },
        "retrieved_juliet": {
            "label": f"Unique retrieved {corpus_label} example",
            "color": "#f59e0b",
            "size": 45,
            "alpha": 0.9,
            "marker": "o",
        },
        "query_cve": {
            "label": "CVE query",
            "color": "#2563eb",
            "size": 70,
            "alpha": 0.95,
            "marker": "^",
        },
    }
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=180)
    for kind, style in groups.items():
        subset = [row for row in rows if row["kind"] == kind]
        if not subset:
            continue
        ax.scatter(
            [float(row["x"]) for row in subset],
            [float(row["y"]) for row in subset],
            s=style["size"],
            c=style["color"],
            alpha=style["alpha"],
            marker=style["marker"],
            label=style["label"],
            edgecolors="#1f2933" if kind != "background_juliet" else "#00664f",
            linewidths=0.5 if kind != "background_juliet" else 0.25,
        )
    for row in rows:
        if row["kind"] == "query_cve":
            ax.annotate(
                f"Q{row['sample_id']}",
                (float(row["x"]), float(row["y"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_title(f"t-SNE of CVE query embeddings and retrieved {corpus_label} examples")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
