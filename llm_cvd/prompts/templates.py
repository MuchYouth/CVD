"""Prompt construction and label parsing for vulnerability classification."""

from __future__ import annotations

import re
from typing import Iterable, Mapping


SYSTEM_PROMPT = (
    "You are a binary vulnerability classifier. "
    "You must output exactly one of these two labels: Vulnerable or Safe. "
    "Do not explain your reasoning. Do not include punctuation, markdown, or extra words."
)


def build_few_shot_prompt(
    examples: Iterable[Mapping[str, object]],
    target_code: str,
) -> str:
    """Build the shared few-shot prompt used for every provider."""
    parts = [
        (
            "Classify the source code into Vulnerable or Safe, and return the "
            "answer as the corresponding label. Here are some examples:"
        )
    ]

    for example in examples:
        parts.extend(
            [
                "",
                "Code:",
                str(example["code"]),
                "Label:",
                str(example["label_text"]),
            ]
        )

    parts.extend(
        [
            "",
            "Code:",
            str(target_code),
            "Answer with exactly one label, either Vulnerable or Safe.",
            "Label:",
        ]
    )

    return "\n".join(parts)


def build_zero_shot_prompt(target_code: str) -> str:
    """Build a direct classification prompt without retrieved examples."""
    parts = [
        (
            "Classify the source code into Vulnerable or Safe, and return the "
            "answer as the corresponding label."
        ),
        "",
        "Code:",
        str(target_code),
        "Answer with exactly one label, either Vulnerable or Safe.",
        "Label:",
    ]

    return "\n".join(parts)


def parse_label(raw_response: str | None) -> str | None:
    """Parse a provider response into Vulnerable/Safe, or None if ambiguous."""
    if raw_response is None:
        return None

    text = raw_response.strip()
    if not text:
        return None

    first_line = text.splitlines()[0].strip()
    first_label = parse_explicit_label(first_line)
    if first_label:
        return first_label

    explicit_label = parse_explicit_label(text)
    if explicit_label:
        return explicit_label

    normalized = normalize_for_label_parse(text[:500])
    if has_safe_signal(normalized) and not has_vulnerable_signal(normalized):
        return "Safe"
    if has_vulnerable_signal(normalized) and not has_safe_signal(normalized):
        return "Vulnerable"

    return None


def parse_explicit_label(text: str) -> str | None:
    """Parse direct label-style answers before falling back to semantic hints."""
    normalized = normalize_for_label_parse(text)
    if not normalized:
        return None

    label_patterns = [
        r"^(?:answer|label|classification|class|result|prediction)\s+"
        r"(?:is\s+)?(?P<label>vulnerable|safe)\b",
        r"^(?P<label>vulnerable|safe)\b",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, normalized)
        if match:
            return label_to_title(match.group("label"))

    matches = re.findall(r"\b(vulnerable|safe)\b", normalized)
    unique_matches = set(matches)
    if len(unique_matches) == 1 and len(matches) == 1:
        return label_to_title(matches[0])
    return None


def normalize_for_label_parse(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z]+", " ", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def label_to_title(label: str) -> str:
    return "Vulnerable" if label == "vulnerable" else "Safe"


def has_safe_signal(text: str) -> bool:
    safe_patterns = [
        r"\bnot vulnerable\b",
        r"\bno (?:clear |obvious |direct |apparent )?vulnerabilit(?:y|ies)\b",
        r"\bdoes not (?:appear to )?(?:contain|have|show) "
        r"(?:a |any )?vulnerabilit(?:y|ies)\b",
        r"\bappears? to be safe\b",
        r"\bthis code is safe\b",
        r"\bbenign\b",
        r"\bclean\b",
    ]
    return any(re.search(pattern, text) for pattern in safe_patterns)


def has_vulnerable_signal(text: str) -> bool:
    vulnerable_patterns = [
        r"\b(?:identify|contains?|has|shows?|demonstrates?|reveals?) "
        r"(?:several |multiple |a |an |potential )?"
        r"(?:security )?vulnerabilit(?:y|ies)\b",
        r"\b(?:is|appears? to be) vulnerable\b",
        r"\b(?:command injection|format string|buffer overflow|sql injection|"
        r"use after free|integer overflow|null pointer dereference) "
        r"(?:vulnerabilit(?:y|ies)|risk|issue)?\b",
        r"\bsecurity (?:bug|flaw|issue|risk)\b",
        r"\bconcerning patterns\b",
        r"\buser controlled data\b",
    ]
    return any(re.search(pattern, text) for pattern in vulnerable_patterns)
