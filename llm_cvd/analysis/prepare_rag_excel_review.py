"""Build an Excel-oriented RAG top-k qualitative review sheet.

This script keeps the original retrieval rows intact, adds side-by-side query
and retrieved trace interpretation columns, converts the existing automatic
yes/partial/no analysis into O/△/X labels, and writes a small per-query summary.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_REVIEW_CSV = "results/rag_retrieval_qualitative_review_cve_k6.csv"
DEFAULT_ANALYSIS_CSV = "results/rag_retrieval_qualitative_review_cve_k6_initial_analysis.csv"
DEFAULT_INSPECTION_CSV = "results/rag_retrieval_inspection_cve_k6.csv"
DEFAULT_RETRIEVED_DETAILS_CSV = "results/retrieved_example_type_details_cve_k6.csv"
DEFAULT_QUERY_META_CSV = "results/rag_query_metadata_template_cve_k6.csv"
DEFAULT_OUTPUT_CSV = "results/rag_retrieval_qualitative_review_cve_k6_excel_review.csv"
DEFAULT_SUMMARY_CSV = "results/rag_retrieval_qualitative_review_cve_k6_excel_summary.csv"

EXCEL_FIELDS = [
    "query_sample_id",
    "query_project",
    "rank",
    "faiss_l2_distance",
    "query_snippet",
    "query_구문",
    "query_의미",
    "query_CWE",
    "query_source_sink",
    "retrieved_snippet_meta",
    "retrieved_snippet",
    "retrieved_구문",
    "retrieved_의미",
    "retrieved_CWE",
    "retrieved_source_sink",
    "구문_유사성",
    "의미_유사성",
    "source_sink_일치",
    "CWE_일치",
    "note",
]

QUERY_META_FIELDS = [
    "query_sample_id",
    "query_project",
    "query_CWE",
    "query_source_sink",
    "query_구문",
    "query_의미",
]

SUMMARY_FIELDS = [
    "query_sample_id",
    "query_project",
    "top_k",
    "구문_유사성_O",
    "구문_유사성_△",
    "구문_유사성_X",
    "의미_유사성_O",
    "의미_유사성_△",
    "의미_유사성_X",
    "source_sink_일치_O",
    "source_sink_일치_△",
    "source_sink_일치_X",
    "CWE_일치_O",
    "CWE_일치_△",
    "CWE_일치_X",
    "CWE_일치_미입력",
    "overall_note",
]

YES_PARTIAL_NO_TO_KO = {
    "yes": "O",
    "partial": "△",
    "no": "X",
    "": "",
}

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CWE_RE = re.compile(r"CWE-?(\d+)")
SOURCE_SINK_STOPWORDS = {
    "source",
    "sink",
    "pattern",
    "input",
    "data",
    "file",
    "string",
    "command",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--analysis-csv", default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--inspection-csv", default=DEFAULT_INSPECTION_CSV)
    parser.add_argument("--retrieved-details-csv", default=DEFAULT_RETRIEVED_DETAILS_CSV)
    parser.add_argument("--query-meta-csv", default=DEFAULT_QUERY_META_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_rows = read_rows(Path(args.review_csv))
    if not review_rows:
        raise ValueError(f"No rows found in {args.review_csv}")

    analysis_by_key = keyed_rows(read_rows(Path(args.analysis_csv)))
    inspection_by_key = keyed_rows(read_rows(Path(args.inspection_csv)))
    retrieved_details = load_retrieved_details(Path(args.retrieved_details_csv))
    query_meta_path = Path(args.query_meta_csv)
    query_meta = load_query_meta(query_meta_path)

    output_rows = []
    for row in review_rows:
        key = row_key(row)
        inspection = inspection_by_key.get(key, {})
        analysis = analysis_by_key.get(key, {})
        retrieved_sample_id = inspection.get("retrieved_sample_id", "")
        detail = retrieved_details.get(retrieved_sample_id, {})
        meta_text = retrieved_meta_text(row, inspection, detail)
        query_info = query_meta.get(row.get("query_sample_id", ""), {})
        retrieved_cwe = extract_cwe(detail.get("cwe_or_path_hint", "") or meta_text)
        query_cwe = query_info.get("query_CWE", "")

        syntax_similarity = label_to_ko(analysis.get("cause_match", ""))
        semantic_similarity = label_to_ko(analysis.get("context_match", ""))
        source_sink_match = infer_source_sink_match(query_info, detail, analysis)
        cwe_match = infer_cwe_match(query_cwe, retrieved_cwe)

        note = build_note(
            analysis.get("note", ""),
            query_cwe=query_cwe,
            retrieved_cwe=retrieved_cwe,
            cwe_match=cwe_match,
            source_sink_match=source_sink_match,
        )

        output_rows.append(
            {
                "query_sample_id": row.get("query_sample_id", ""),
                "query_project": row.get("query_project", ""),
                "rank": row.get("rank", ""),
                "faiss_l2_distance": row.get("faiss_l2_distance", ""),
                "query_snippet": row.get("query_snippet", ""),
                "query_구문": query_info.get("query_구문", summarize_syntax(row.get("query_snippet", ""))),
                "query_의미": query_info.get("query_의미", "수동 입력 필요: query trace의 취약 흐름 요약"),
                "query_CWE": query_cwe,
                "query_source_sink": query_info.get("query_source_sink", ""),
                "retrieved_snippet_meta": meta_text,
                "retrieved_snippet": row.get("retrieved_snippet", ""),
                "retrieved_구문": summarize_retrieved_syntax(
                    row.get("retrieved_snippet", ""),
                    detail,
                ),
                "retrieved_의미": summarize_retrieved_meaning(detail),
                "retrieved_CWE": retrieved_cwe,
                "retrieved_source_sink": infer_retrieved_source_sink(detail),
                "구문_유사성": syntax_similarity,
                "의미_유사성": semantic_similarity,
                "source_sink_일치": source_sink_match,
                "CWE_일치": cwe_match,
                "note": note,
            }
        )

    write_rows(Path(args.output_csv), EXCEL_FIELDS, output_rows)
    write_query_meta_template(query_meta_path, review_rows, query_meta)
    write_rows(Path(args.summary_csv), SUMMARY_FIELDS, build_summary_rows(output_rows))

    print(f"Wrote Excel review CSV: {args.output_csv}")
    print(f"Wrote query metadata template: {args.query_meta_csv}")
    print(f"Wrote per-query summary CSV: {args.summary_csv}")
    print("Fill query metadata, rerun this script, then review O/△/X labels in Excel.")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def keyed_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {row_key(row): row for row in rows}


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("query_sample_id", ""), row.get("rank", "")


def load_retrieved_details(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    return {row.get("retrieved_sample_id", ""): row for row in rows}


def load_query_meta(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    return {row.get("query_sample_id", ""): row for row in rows if row.get("query_sample_id", "")}


def write_query_meta_template(
    path: Path,
    review_rows: list[dict[str, str]],
    existing_meta: dict[str, dict[str, str]],
) -> None:
    seen = set()
    template_rows = []
    for row in review_rows:
        query_id = row.get("query_sample_id", "")
        if query_id in seen:
            continue
        seen.add(query_id)
        meta = existing_meta.get(query_id, {})
        template_rows.append(
            {
                "query_sample_id": query_id,
                "query_project": row.get("query_project", ""),
                "query_CWE": meta.get("query_CWE", ""),
                "query_source_sink": meta.get("query_source_sink", ""),
                "query_구문": meta.get("query_구문", summarize_syntax(row.get("query_snippet", ""))),
                "query_의미": meta.get("query_의미", ""),
            }
        )
    write_rows(path, QUERY_META_FIELDS, template_rows)


def retrieved_meta_text(
    review_row: dict[str, str],
    inspection_row: dict[str, str],
    detail: dict[str, str],
) -> str:
    existing_meta = review_row.get("cause_match", "")
    if existing_meta and ("|" in existing_meta or "CWE" in existing_meta):
        return existing_meta

    parts = [
        inspection_row.get("retrieved_sample_id", ""),
        f"file_name={detail.get('file_name', '')}" if detail.get("file_name", "") else "",
        detail.get("cwe_or_path_hint", ""),
    ]
    return " | ".join(part for part in parts if part)


def label_to_ko(value: str) -> str:
    return YES_PARTIAL_NO_TO_KO.get(value.strip().lower(), value)


def extract_cwe(text: str) -> str:
    match = CWE_RE.search(text)
    return f"CWE-{match.group(1)}" if match else ""


def infer_retrieved_source_sink(detail: dict[str, str]) -> str:
    hint = detail.get("cwe_or_path_hint", "")
    if "__" not in hint:
        return ""
    tail = hint.rsplit("__", 1)[-1]
    parts = [part for part in tail.split("_") if part]
    if len(parts) >= 3 and parts[0] in {"char", "wchar_t"}:
        return f"source: {parts[1]} / sink: {'_'.join(parts[2:])}"
    if len(parts) >= 2:
        return f"source: {parts[0]} / sink: {'_'.join(parts[1:])}"
    if parts:
        return f"pattern: {parts[0]}"
    return ""


def summarize_retrieved_syntax(snippet: str, detail: dict[str, str]) -> str:
    hint = detail.get("cwe_or_path_hint", "")
    source_sink = infer_retrieved_source_sink(detail)

    if "CWE78" in hint:
        source = source_sink_value(source_sink, "source") or "외부 입력"
        return (
            f"{source}에서 온 문자열을 data 값으로 전달하고, 문자열 결합 또는 객체 호출 체인을 거친 뒤, "
            "system() 호출의 명령 문자열 인자로 사용하는 구문"
        )

    if "fscanf" in hint and any(name in hint for name in ["strncpy", "memcpy", "memmove"]):
        sink = source_sink_value(source_sink, "sink") or buffer_copy_sink(snippet) or "버퍼 복사 함수"
        return (
            f"fscanf()로 읽은 정수 값을 변수에 저장하고, 그 값을 함수 반환 또는 포인터 전달로 유지한 뒤, "
            f"범위 검사 후 {sink}() 호출의 길이 인자로 사용하는 구문"
        )

    if "fgets" in hint or "fscanf" in hint:
        source = source_sink_value(source_sink, "source") or ("fgets" if "fgets" in hint else "fscanf")
        return (
            f"{source}()로 읽은 값을 정수 count 변수에 저장하고, 그 값을 포인터 또는 지역 변수로 전달한 뒤, "
            "조건 검사 후 fwrite() 반복/크기 인자로 사용하는 구문"
        )

    if "rand" in hint and "fwrite" in hint:
        return (
            "RAND32()로 생성한 정수 값을 count 변수에 대입하고, 조건 분기와 범위 검사를 거친 뒤, "
            "fwrite() 호출의 반복/크기 인자로 사용하는 구문"
        )

    if "rand" in hint and any(name in hint for name in ["strncpy", "memcpy", "memmove"]):
        sink = source_sink_value(source_sink, "sink") or buffer_copy_sink(snippet) or "버퍼 복사 함수"
        return (
            f"RAND32()로 생성한 정수 값을 data 변수에 대입하고, 조건 분기와 범위 검사를 거친 뒤, "
            f"{sink}() 호출의 길이 인자로 사용하는 구문"
        )

    return summarize_syntax(snippet)


def summarize_retrieved_meaning(detail: dict[str, str]) -> str:
    hint = detail.get("cwe_or_path_hint", "")
    cwe = extract_cwe(hint)
    source_sink = infer_retrieved_source_sink(detail)
    source = source_sink_value(source_sink, "source")
    sink = source_sink_value(source_sink, "sink")

    if cwe == "CWE-78":
        source_text = f"{source} 입력" if source else "외부 영향 문자열"
        sink_text = f"{sink} 호출" if sink else "system() 호출"
        return (
            f"{source_text}이 shell command 문자열에 포함되어 {sink_text}에서 "
            "OS command injection으로 이어질 수 있는 취약 흐름"
        )

    if cwe == "CWE-195":
        sink_text = f"{sink}() 길이 인자" if sink else "버퍼 처리 함수의 길이 인자"
        return (
            f"음수 또는 큰 signed 정수 값이 {sink_text}로 전달되면서 unsigned 값으로 해석되어 "
            "과도한 복사나 메모리 손상으로 이어질 수 있는 취약 흐름"
        )

    if cwe == "CWE-194":
        sink_text = f"{sink}() 길이 인자" if sink else "버퍼 처리 함수의 길이 인자"
        return (
            f"short 등 작은 정수 타입의 값이 부호 확장된 뒤 {sink_text}로 전달되어 "
            "예상보다 큰 길이로 처리될 수 있는 취약 흐름"
        )

    if cwe == "CWE-400":
        source_text = f"{source} 입력값" if source else "외부 영향 count 값"
        sink_text = f"{sink}()" if sink else "fwrite()"
        return (
            f"{source_text}이 반복 횟수나 출력 크기 제어값으로 사용되어 {sink_text} 호출에서 "
            "resource exhaustion으로 이어질 수 있는 취약 흐름"
        )

    return detail.get("retrieved_type", "retrieved trace 의미 요약 필요")


def source_sink_value(source_sink: str, key: str) -> str:
    for part in source_sink.split("/"):
        name, sep, value = part.strip().partition(":")
        if sep and name.strip() == key:
            return value.strip()
    return ""


def buffer_copy_sink(snippet: str) -> str:
    for name in ["strncpy", "memcpy", "memmove"]:
        if f"{name}(" in snippet or f"{name} (" in snippet:
            return name
    return ""


def infer_source_sink_match(
    query_info: dict[str, str],
    detail: dict[str, str],
    analysis: dict[str, str],
) -> str:
    query_source_sink = query_info.get("query_source_sink", "").strip()
    retrieved_source_sink = infer_retrieved_source_sink(detail)
    if not query_source_sink or not retrieved_source_sink:
        return "미입력"
    query_lower = query_source_sink.lower()
    retrieved_lower = retrieved_source_sink.lower()
    tokens = {
        token
        for token in re.split(r"[^a-z0-9_]+", query_lower)
        if len(token) >= 3 and token not in SOURCE_SINK_STOPWORDS
    }
    if tokens and any(token in retrieved_lower for token in tokens):
        return "O"
    return label_to_ko(analysis.get("context_match", "")) or "X"


def infer_cwe_match(query_cwe: str, retrieved_cwe: str) -> str:
    if not query_cwe or not retrieved_cwe:
        return "미입력"
    query_values = {f"CWE-{value}" for value in CWE_RE.findall(query_cwe)}
    retrieved_values = {f"CWE-{value}" for value in CWE_RE.findall(retrieved_cwe)}
    return "O" if query_values & retrieved_values else "X"


def summarize_syntax(snippet: str, limit: int = 6) -> str:
    calls = []
    for call in CALL_RE.findall(snippet):
        if call in {"if", "for", "while", "switch", "return", "sizeof"}:
            continue
        if call not in calls:
            calls.append(call)
        if len(calls) == limit:
            break
    if calls:
        return "핵심 호출: " + ", ".join(calls)

    operators = []
    for operator in ["->", "=", "<", ">", "+", "-", "*", "/"]:
        if operator in snippet:
            operators.append(operator)
    if operators:
        return "핵심 연산: " + ", ".join(operators[:limit])
    return "구문 요약 필요"


def build_note(
    analysis_note: str,
    query_cwe: str,
    retrieved_cwe: str,
    cwe_match: str,
    source_sink_match: str,
) -> str:
    note = analysis_note.strip()
    if not note:
        note = "자동 분석 근거 없음; 사람이 최종 검토 필요."
    additions = []
    if not query_cwe:
        additions.append("query CWE 수동 입력 필요")
    if not retrieved_cwe:
        additions.append("retrieved CWE 확인 필요")
    if cwe_match == "O":
        additions.append("CWE 일치")
    elif cwe_match == "X":
        additions.append("CWE 불일치")
    if source_sink_match == "미입력":
        additions.append("source/sink 수동 입력 필요")
    if additions:
        note = f"{note} ({'; '.join(additions)})"
    return note


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_sample_id"]].append(row)

    summary_rows = []
    for query_id in sorted(grouped, key=lambda value: int(value) if value.isdigit() else value):
        group = sorted(grouped[query_id], key=lambda row: int(row["rank"]))
        first = group[0]
        syntax = Counter(row["구문_유사성"] for row in group)
        semantic = Counter(row["의미_유사성"] for row in group)
        source_sink = Counter(row["source_sink_일치"] for row in group)
        cwe = Counter(row["CWE_일치"] for row in group)
        helpful_candidates = sum(
            1
            for row in group
            if row["구문_유사성"] in {"O", "△"}
            or row["의미_유사성"] in {"O", "△"}
            or row["source_sink_일치"] in {"O", "△"}
            or row["CWE_일치"] == "O"
        )
        summary_rows.append(
            {
                "query_sample_id": query_id,
                "query_project": first["query_project"],
                "top_k": str(len(group)),
                "구문_유사성_O": str(syntax["O"]),
                "구문_유사성_△": str(syntax["△"]),
                "구문_유사성_X": str(syntax["X"]),
                "의미_유사성_O": str(semantic["O"]),
                "의미_유사성_△": str(semantic["△"]),
                "의미_유사성_X": str(semantic["X"]),
                "source_sink_일치_O": str(source_sink["O"]),
                "source_sink_일치_△": str(source_sink["△"]),
                "source_sink_일치_X": str(source_sink["X"]),
                "CWE_일치_O": str(cwe["O"]),
                "CWE_일치_△": str(cwe["△"]),
                "CWE_일치_X": str(cwe["X"]),
                "CWE_일치_미입력": str(cwe["미입력"]),
                "overall_note": f"top-6 중 자동 기준상 일부라도 관련 있는 후보 {helpful_candidates}개",
            }
        )
    return summary_rows


if __name__ == "__main__":
    main()
