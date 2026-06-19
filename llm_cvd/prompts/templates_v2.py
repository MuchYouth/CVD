"""Trace-aware prompt construction for RAG v2."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping

from llm_cvd.prompts.templates import SYSTEM_PROMPT, parse_label


def build_trace_aware_prompt(
    examples: Iterable[Mapping[str, Any]],
    target_code: str,
    target_trace: Mapping[str, Any] | object | None = None,
) -> str:
    """Build a structured trace-aware few-shot prompt."""
    parts = [
        (
            "Classify the target source code into Vulnerable or Safe. "
            "Use the reference traces as structural hints, not as ground truth for the target."
        ),
    ]

    for index, example in enumerate(examples, start=1):
        trace = trace_mapping(example.get("trace_features_v2"))
        parts.extend(
            [
                "",
                f"Reference Trace {index}",
                f"CWE: {field_value(example, trace, 'cwe_hint', 'cwe', default='UNKNOWN')}",
                f"Label: {example.get('label_text', '')}",
                f"Source: {field_value(example, trace, 'source_hint', default='unknown source')}",
                f"Flow: {field_value(example, trace, 'flow_hint', default='unknown flow')}",
                f"Sink: {field_value(example, trace, 'sink_hint', default='unknown sink')}",
                f"Root cause hint: {field_value(example, trace, 'root_cause_hint', default='insufficient trace evidence')}",
                f"Abstract trace: {field_value(example, trace, 'abstract_trace', default='[TRACE:unknown]')}",
            ]
        )

    target_trace_map = trace_mapping(target_trace)
    if target_trace_map:
        parts.extend(
            [
                "",
                "Target Trace Hint",
                f"CWE: {target_trace_map.get('cwe') or 'UNKNOWN'}",
                f"Source: {target_trace_map.get('source_hint') or 'unknown source'}",
                f"Flow: {target_trace_map.get('flow_hint') or 'unknown flow'}",
                f"Sink: {target_trace_map.get('sink_hint') or 'unknown sink'}",
                f"Abstract trace: {target_trace_map.get('abstract_trace') or '[TRACE:unknown]'}",
            ]
        )

    parts.extend(
        [
            "",
            "Target Code:",
            str(target_code),
            "",
            "Decision procedure:",
            "1. Identify the target source.",
            "2. Check whether value or influence flows from the source to a sink.",
            "3. Decide whether the sink performs a dangerous operation.",
            "4. Check for sanitizer, validation, escaping, fixed format string, or bounds check evidence.",
            "5. Compare the target vulnerability-flow structure against the reference traces.",
            "Answer with exactly one label, either Vulnerable or Safe.",
            "Label:",
        ]
    )
    return "\n".join(parts)


def trace_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return data if isinstance(data, dict) else {}
    return {}


def field_value(
    example: Mapping[str, Any],
    trace: Mapping[str, Any],
    *keys: str,
    default: str,
) -> str:
    for key in keys:
        value = example.get(key)
        if value:
            return str(value)
        value = trace.get(key)
        if value:
            if isinstance(value, list):
                return ", ".join(str(item) for item in value)
            return str(value)
    return default
