"""Trace abstraction helpers for trace-aware RAG v2 experiments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from llm_cvd.data.juliet_loader import parse_real_vul_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH_PREFIX = "tracer-signaturedb-rag/signature-db-ori/"

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "case", "else"}
CALL_EXCLUDE = CONTROL_KEYWORDS | {
    "return",
    "sizeof",
    "typedef",
    "struct",
    "class",
    "catch",
}

SOURCE_API_GROUPS = {
    "socket": {"recv", "recvfrom", "read", "accept"},
    "file": {"fgets", "fscanf", "scanf", "fread", "read", "getc", "fgetc"},
    "env": {"getenv"},
    "console": {"scanf", "gets"},
    "argument": {"argv", "optarg"},
}
SINK_API_GROUPS = {
    "command": {"system", "popen", "execl", "execlp", "execle", "execv", "execvp"},
    "format": {"printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "vsprintf", "vsnprintf"},
    "memory": {"strcpy", "strcat", "memcpy", "memmove", "malloc", "new", "free", "delete"},
    "integer": {"malloc", "calloc", "realloc", "new"},
}
FLOW_API_GROUPS = {
    "string": {"strlen", "strchr", "strncpy", "strncat", "strcmp", "strcpy", "strcat", "memcpy", "memset"},
    "conversion": {"atoi", "atol", "strtol", "strtoul", "sscanf"},
    "library_call": set(),
}
SANITIZER_API_GROUPS = {
    "validation": {"isalnum", "isdigit", "isalpha", "strcmp", "strncmp", "strlen"},
    "bounds": {"snprintf", "strncpy", "strncat", "sizeof"},
    "escaping": {"escape", "quote", "sanitize"},
}

ALL_SOURCE_APIS = set().union(*SOURCE_API_GROUPS.values())
ALL_SINK_APIS = set().union(*SINK_API_GROUPS.values())
ALL_FLOW_APIS = set().union(*FLOW_API_GROUPS.values())
ALL_SANITIZER_APIS = set().union(*SANITIZER_API_GROUPS.values())


@dataclass
class TraceFeatures:
    """Serializable trace abstraction used by retrieval and prompt construction."""

    cwe: str = ""
    label: str = ""
    source_apis: set[str] = field(default_factory=set)
    source_kinds: set[str] = field(default_factory=set)
    sink_apis: set[str] = field(default_factory=set)
    sink_kinds: set[str] = field(default_factory=set)
    flow_apis: set[str] = field(default_factory=set)
    flow_kinds: set[str] = field(default_factory=set)
    controls: set[str] = field(default_factory=set)
    sanitizer_apis: set[str] = field(default_factory=set)
    sanitizer_kinds: set[str] = field(default_factory=set)
    normalized_tokens: set[str] = field(default_factory=set)
    abstract_trace: str = ""
    source_hint: str = "unknown source"
    flow_hint: str = "unknown flow"
    sink_hint: str = "unknown sink"
    root_cause_hint: str = "insufficient trace evidence"
    source_signature_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, set):
                data[key] = sorted(value)
        return data


class TraceAbstractorV2:
    """Build compact source/flow/sink features from records and optional trace JSON."""

    def __init__(self, csv_paths: Iterable[str | Path] | None = None) -> None:
        self.raw_rows_by_id: dict[str, dict[str, str]] = {}
        for csv_path in csv_paths or []:
            self.add_csv(csv_path)

    def add_csv(self, csv_path: str | Path | None) -> None:
        if not csv_path:
            return
        path = Path(csv_path)
        if not path.exists() or not path.is_file():
            return
        try:
            raw_rows = parse_real_vul_csv(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            return
        for row in raw_rows:
            for key in row_keys(row):
                self.raw_rows_by_id.setdefault(key, row)

    def abstract_record(self, record: dict[str, Any]) -> TraceFeatures:
        raw_row = self.raw_rows_by_id.get(sample_id_value(record), {})
        merged = {**raw_row, **record}
        code = str(merged.get("code") or merged.get("processed_func") or "")
        trace_path = str(raw_row.get("source_signature_path", ""))
        trace_json = load_trace_json(trace_path)

        features = TraceFeatures(
            cwe=extract_cwe_hint(" ".join(str(merged.get(key, "")) for key in ("cwe", "project", "source", "source_signature_path", "code", "processed_func"))),
            label=str(merged.get("label_text", "")),
            source_signature_path=trace_path,
        )
        add_code_features(features, code)
        if trace_json:
            add_bug_trace_features(features, trace_json)
        finalize_features(features)
        return features

    def enrich_record(self, record: dict[str, Any]) -> dict[str, Any]:
        features = self.abstract_record(record)
        enriched = dict(record)
        enriched["trace_features_v2"] = features.to_dict()
        enriched["abstract_trace"] = features.abstract_trace
        enriched["source_hint"] = features.source_hint
        enriched["flow_hint"] = features.flow_hint
        enriched["sink_hint"] = features.sink_hint
        enriched["root_cause_hint"] = features.root_cause_hint
        enriched["cwe_hint"] = features.cwe
        return enriched


def row_keys(row: dict[str, str]) -> set[str]:
    keys = set()
    unique_id = str(row.get("unique_id", "")).strip()
    file_name = str(row.get("file_name", "")).strip()
    if unique_id:
        keys.update({unique_id, f"trace-csv::{unique_id}"})
    if file_name:
        keys.add(file_name)
    return keys


def sample_id_value(record: dict[str, Any]) -> str:
    return str(record.get("sample_id", "")).strip()


def load_trace_json(source_signature_path: str) -> dict[str, Any] | None:
    for path in candidate_trace_paths(source_signature_path):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def candidate_trace_paths(source_signature_path: str) -> list[Path]:
    raw = str(source_signature_path or "").strip()
    if not raw:
        return []
    raw_path = Path(raw)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(PROJECT_ROOT / raw_path)
        if TRACE_PATH_PREFIX in raw:
            suffix = raw.split(TRACE_PATH_PREFIX, 1)[1]
            candidates.append(PROJECT_ROOT / "llm_cvd" / "traceConvert" / "signature-db-ori" / suffix)
            candidates.append(PROJECT_ROOT / "dataset" / "signature-db" / suffix)
        if "signature-db-ori/" in raw:
            suffix = raw.split("signature-db-ori/", 1)[1]
            candidates.append(PROJECT_ROOT / "llm_cvd" / "traceConvert" / "signature-db-ori" / suffix)
        if "signature-db/" in raw:
            suffix = raw.split("signature-db/", 1)[1]
            candidates.append(PROJECT_ROOT / "dataset" / "signature-db" / suffix)
    deduped = []
    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def add_code_features(features: TraceFeatures, code: str) -> None:
    normalized = normalize_code(code)
    features.normalized_tokens.update(token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", normalized))
    calls = extract_calls(code)
    identifiers = set(calls) | set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", code))
    lowered = {value.lower() for value in identifiers}

    for api in sorted(lowered & ALL_SOURCE_APIS):
        features.source_apis.add(api)
        features.source_kinds.update(kinds_for_api(api, SOURCE_API_GROUPS))
    for api in sorted(lowered & ALL_SINK_APIS):
        features.sink_apis.add(api)
        features.sink_kinds.update(kinds_for_api(api, SINK_API_GROUPS))
    for api in sorted(lowered & ALL_FLOW_APIS):
        features.flow_apis.add(api)
        features.flow_kinds.update(kinds_for_api(api, FLOW_API_GROUPS))
    for api in sorted(lowered & ALL_SANITIZER_APIS):
        features.sanitizer_apis.add(api)
        features.sanitizer_kinds.update(kinds_for_api(api, SANITIZER_API_GROUPS))
    for keyword in CONTROL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", code):
            features.controls.add(keyword)
            features.flow_kinds.add("control")
    if "*" in code or "->" in code or "&" in code:
        features.flow_kinds.add("pointer")
    if any(api in lowered for api in FLOW_API_GROUPS["string"]):
        features.flow_kinds.add("string")


def add_bug_trace_features(features: TraceFeatures, trace_json: dict[str, Any]) -> None:
    traces = trace_json.get("bug_trace")
    if not isinstance(traces, list):
        return
    best_trace = max((trace for trace in traces if isinstance(trace, list)), key=len, default=[])
    trace_steps = []
    for event in best_trace:
        if not isinstance(event, dict):
            continue
        description = str(event.get("description", "")).lower()
        feature = parse_feature(event.get("feature"))
        kind = str(feature[0]).lower() if feature else description.split(",", 1)[0].strip()
        api = first_api_from_feature(feature) or first_api_from_description(description)
        role = role_for_trace_kind(kind, description)
        if api:
            api = api.lower()
        if role == "SOURCE":
            if api:
                features.source_apis.add(api)
                features.source_kinds.update(kinds_for_api(api, SOURCE_API_GROUPS) or {"input"})
            trace_steps.append(f"[SOURCE:{api or 'input'}]")
        elif role == "SINK":
            if api:
                features.sink_apis.add(api)
                features.sink_kinds.update(kinds_for_api(api, SINK_API_GROUPS) or {kind})
            trace_steps.append(f"[SINK:{api or kind or 'sink'}]")
        else:
            flow_name = "if" if kind == "prune" else api or kind or "call"
            if flow_name == "if":
                features.controls.add("if")
                features.flow_kinds.add("control")
            elif api:
                features.flow_apis.add(api)
                features.flow_kinds.update(kinds_for_api(api, FLOW_API_GROUPS) or {"library_call"})
            trace_steps.append(f"[FLOW:{flow_name}]")
    if trace_steps:
        features.abstract_trace = " -> ".join(dedupe_adjacent(trace_steps))


def parse_feature(raw_feature: Any) -> list[Any]:
    if isinstance(raw_feature, list):
        return raw_feature
    if not isinstance(raw_feature, str) or not raw_feature.strip():
        return []
    try:
        parsed = json.loads(raw_feature)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def first_api_from_feature(feature: list[Any]) -> str:
    for item in feature[1:]:
        if isinstance(item, str) and re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", item):
            return item
    return ""


def first_api_from_description(description: str) -> str:
    for part in description.split(","):
        value = part.strip()
        if re.match(r"^[a-zA-Z_][a-zA-Z_0-9]*$", value):
            return value
    return ""


def role_for_trace_kind(kind: str, description: str) -> str:
    text = f"{kind} {description}".lower()
    if "input" in text:
        return "SOURCE"
    if any(token in text for token in ("injection", "format", "overflow", "free", "uaf", "leak", "null")):
        return "SINK"
    return "FLOW"


def finalize_features(features: TraceFeatures) -> None:
    if not features.abstract_trace:
        steps = []
        if features.source_apis:
            steps.append(f"[SOURCE:{first_sorted(features.source_apis)}]")
        for control in sorted(features.controls):
            steps.append(f"[FLOW:{control}]")
        for flow in sorted(features.flow_apis - features.source_apis - features.sink_apis):
            steps.append(f"[FLOW:{flow}]")
        if not any(step.startswith("[FLOW:") for step in steps) and features.flow_kinds:
            steps.append(f"[FLOW:{first_sorted(features.flow_kinds)}]")
        if features.sink_apis:
            steps.append(f"[SINK:{first_sorted(features.sink_apis)}]")
        features.abstract_trace = " -> ".join(steps) if steps else "[TRACE:unknown]"

    features.source_hint = source_hint(features)
    features.flow_hint = flow_hint(features)
    features.sink_hint = sink_hint(features)
    features.root_cause_hint = root_cause_hint(features)


def source_hint(features: TraceFeatures) -> str:
    if features.source_apis:
        api = first_sorted(features.source_apis)
        kinds = ", ".join(sorted(features.source_kinds)) or "input"
        return f"{kinds} via {api}"
    return "unknown source"


def flow_hint(features: TraceFeatures) -> str:
    parts = []
    if features.controls:
        parts.append("control branch " + "/".join(sorted(features.controls)))
    if features.flow_apis:
        parts.append("library call " + "/".join(sorted(features.flow_apis)))
    if features.flow_kinds:
        parts.append("flow kind " + "/".join(sorted(features.flow_kinds)))
    return "; ".join(parts) if parts else "unknown flow"


def sink_hint(features: TraceFeatures) -> str:
    if features.sink_apis:
        api = first_sorted(features.sink_apis)
        kinds = ", ".join(sorted(features.sink_kinds)) or "dangerous operation"
        return f"{api} ({kinds})"
    return "unknown sink"


def root_cause_hint(features: TraceFeatures) -> str:
    if features.source_apis and features.sink_apis and not features.sanitizer_apis:
        return (
            f"external input from {first_sorted(features.source_apis)} reaches "
            f"{first_sorted(features.sink_apis)} without an obvious sanitizer"
        )
    if features.sink_apis:
        return f"dangerous sink {first_sorted(features.sink_apis)} needs validation or safe usage"
    if features.sanitizer_apis:
        return f"sanitizer/fix evidence present via {first_sorted(features.sanitizer_apis)}"
    return "insufficient trace evidence"


def normalize_code(code: str) -> str:
    text = re.sub(r'"(?:\\.|[^"\\])*"', "<STR>", code)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "<STR>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", text)
    calls = set(extract_calls(text))
    for call in sorted(calls, key=len, reverse=True):
        if call not in CALL_EXCLUDE:
            text = re.sub(rf"\b{re.escape(call)}\b(?=\s*\()", "<FUNC>", text)
    return re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*\b", "<VAR>", text)


def extract_calls(code: str) -> list[str]:
    return [
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", code)
        if match.group(1).lower() not in CALL_EXCLUDE
    ]


def kinds_for_api(api: str, groups: dict[str, set[str]]) -> set[str]:
    return {kind for kind, values in groups.items() if api in values}


def extract_cwe_hint(text: str) -> str:
    values = sorted({f"CWE-{match}" for match in re.findall(r"CWE[-_]?(\d+)", text, re.I)})
    return ";".join(values)


def first_sorted(values: set[str]) -> str:
    return sorted(values)[0] if values else ""


def dedupe_adjacent(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result
