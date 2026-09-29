"""Structured-output validation unit tests."""

from __future__ import annotations

import pytest

from blindllm.contracts import ALLOWED_TOOL_NAMES
from blindllm.validation import validate_model_output


def test_valid_final_answer():
    validate_model_output(
        {
            "response_type": "final_answer",
            "tool_name": None,
            "tool_args": None,
            "narrative": "ok",
            "citations": None,
        }
    )


def test_valid_tool_call_with_allowed_tool():
    validate_model_output(
        {
            "response_type": "tool_call",
            "tool_name": "list_schemas",
            "tool_args": {"dataset": None, "schema": None, "filter": None},
            "narrative": None,
            "citations": None,
        }
    )


def test_missing_response_type_raises():
    with pytest.raises(ValueError, match="response_type"):
        validate_model_output({})


def test_unsupported_response_type_raises():
    with pytest.raises(ValueError, match="unsupported response_type"):
        validate_model_output({"response_type": "ohai"})


def test_disallowed_tool_name_raises():
    with pytest.raises(ValueError, match="tool_name not allowed"):
        validate_model_output(
            {
                "response_type": "tool_call",
                "tool_name": "not_a_tool",
                "tool_args": {"dataset": None, "schema": None, "filter": None},
                "narrative": None,
                "citations": None,
            }
        )


def test_tool_args_must_be_dict():
    with pytest.raises(ValueError, match="tool_args must be an object"):
        validate_model_output(
            {
                "response_type": "tool_call",
                "tool_name": "list_schemas",
                "tool_args": None,
                "narrative": None,
                "citations": None,
            }
        )


def test_empty_narrative_for_final_answer_raises():
    with pytest.raises(ValueError, match="narrative must be a non-empty string"):
        validate_model_output(
            {
                "response_type": "final_answer",
                "tool_name": None,
                "tool_args": None,
                "narrative": "   ",
                "citations": None,
            }
        )


def test_unexpected_fields_raise():
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_model_output(
            {
                "response_type": "final_answer",
                "tool_name": None,
                "tool_args": None,
                "narrative": "ok",
                "citations": None,
                "extra_field": "should not be here",
            }
        )


def test_allowed_tools_set_matches_waterfall_surface():
    assert {
        "list_schemas",
        "describe_schema",
        "query_aggregate",
        "suggest_ml_approach",
    } == ALLOWED_TOOL_NAMES
