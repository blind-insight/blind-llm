"""Leakage policy: PII regex + aggregate-only payload gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PII_PATTERN = re.compile(
    r"\b(\d{3}-\d{2}-\d{4}|\d{9}|[A-Z]{2}\d{6,}|[^\s]+@[^\s]+\.[^\s]+)\b"
)


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    reason: str


def validate_tool_payload(payload: dict[str, Any]) -> PolicyDecision:
    if "raw_records" in payload:
        return PolicyDecision(False, "raw_records are not allowed in tool payload")

    if "identifiers" in payload:
        return PolicyDecision(
            False, "direct identifiers are not allowed in tool payload"
        )

    if payload.get("contains_pii") is True:
        return PolicyDecision(False, "payload flagged as containing PII")

    for key, value in payload.items():
        if isinstance(value, str) and PII_PATTERN.search(value):
            return PolicyDecision(False, f"PII-like token detected in '{key}'")

    return PolicyDecision(True, "payload is aggregate-only and alias-safe")


def validate_narrative(text: str) -> PolicyDecision:
    if PII_PATTERN.search(text):
        return PolicyDecision(False, "PII-like token detected in narrative")
    return PolicyDecision(True, "narrative passes leakage checks")
