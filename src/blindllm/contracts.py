"""Analyst contracts: tool name set and OpenAI structured-output schema.

Schema selection is LLM-driven. The orchestrator exposes three generic
tools and lets the model pick which dataset(s)/schema(s) to query based
on the user's question plus an optional `@<slug>` hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALLOWED_TOOL_NAMES: set[str] = {
    "list_schemas",
    "describe_schema",
    "query_aggregate",
    "suggest_ml_approach",
}

OPENAI_STRUCTURED_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response_type": {
            "type": "string",
            "enum": ["tool_call", "final_answer"],
        },
        "tool_name": {
            "type": ["string", "null"],
            "enum": [*sorted(ALLOWED_TOOL_NAMES), None],
        },
        "tool_args": {
            "type": ["object", "null"],
            "properties": {
                "dataset": {"type": ["string", "null"]},
                "schema": {"type": ["string", "null"]},
                "filter": {"type": ["string", "null"]},
            },
            "required": ["dataset", "schema", "filter"],
            "additionalProperties": False,
        },
        "narrative": {"type": ["string", "null"]},
        "citations": {"type": ["array", "null"], "items": {"type": "string"}},
    },
    "required": ["response_type", "tool_name", "tool_args", "narrative", "citations"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class AnalystRequest:
    user_id: str
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    provider_user_id: str | None = None


@dataclass(slots=True)
class OrchestratorResponse:
    answer: str
    used_tool: str | None
    tool_payload: dict[str, Any] | None
    safe: bool
    fallback: bool
    fallback_reason: str | None = None
    data_mode: str = "mock"
    schemas_queried: list[str] = field(default_factory=list)
    provider_error: dict[str, Any] | None = None
