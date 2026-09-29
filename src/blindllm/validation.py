"""Validate the structured-output payload returned by the LLM."""

from __future__ import annotations

from typing import Any

from blindllm.contracts import ALLOWED_TOOL_NAMES

_EXPECTED_KEYS = {"response_type", "tool_name", "tool_args", "narrative", "citations"}


def validate_model_output(
    payload: dict[str, Any],
    allowed_tools: set[str] | None = None,
) -> None:
    if allowed_tools is None:
        allowed_tools = ALLOWED_TOOL_NAMES

    if "response_type" not in payload:
        raise ValueError("model output missing required field: response_type")

    response_type = payload["response_type"]
    if response_type not in {"tool_call", "final_answer"}:
        raise ValueError(f"unsupported response_type: {response_type}")

    extra_fields = set(payload.keys()) - _EXPECTED_KEYS
    if extra_fields:
        raise ValueError(f"unexpected fields in model output: {sorted(extra_fields)}")

    if response_type == "tool_call":
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args")
        if tool_name not in allowed_tools:
            raise ValueError(f"tool_name not allowed: {tool_name}")
        if not isinstance(tool_args, dict):
            raise ValueError("tool_args must be an object for tool_call")

    if response_type == "final_answer":
        narrative = payload.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative must be a non-empty string for final_answer")
