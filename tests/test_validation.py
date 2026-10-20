"""Structured-output validation unit tests."""

from __future__ import annotations

import pytest

from blindllm.catalog import MOCK_CATALOG
from blindllm.contracts import ALLOWED_TOOL_NAMES, OPENAI_STRUCTURED_OUTPUT_SCHEMA
from blindllm.filters import PhraseValidationError
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
            "tool_args": {"dataset": None, "schema": None, "filters": None},
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
                "tool_args": {"dataset": None, "schema": None, "filters": None},
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


def _query(filters, dataset="population-health", schema="ehr-data-v3"):
    return {
        "response_type": "tool_call",
        "tool_name": "query_aggregate",
        "tool_args": {"dataset": dataset, "schema": schema, "filters": filters},
        "narrative": None,
        "citations": None,
    }


MRI_BY_ER = [
    {"field": "procedure", "op": "phrase", "value": "MRI"},
    {"field": "department", "op": "eq", "value": "emergency"},
    {"field": "year", "op": "eq", "value": "2025"},
    {"field": "month", "op": "eq", "value": "3"},
    {"field": "patient_id", "op": "count", "value": "100001~139385"},
]


def test_query_aggregate_with_phrase_filters_validates_against_catalog():
    validate_model_output(_query(MRI_BY_ER), catalog=MOCK_CATALOG)


def test_query_aggregate_rejects_legacy_filter_string():
    payload = _query(None)
    payload["tool_args"] = {"dataset": "d", "schema": "s", "filter": "a:b"}
    with pytest.raises(ValueError, match="structured `filters` list"):
        validate_model_output(payload)


def test_query_aggregate_rejects_zero_token_phrase():
    with pytest.raises(PhraseValidationError, match="zero tokens"):
        validate_model_output(
            _query([{"field": "note", "op": "phrase", "value": "???"}])
        )


def test_query_aggregate_rejects_eq_on_text_field_with_catalog():
    with pytest.raises(ValueError, match="not supported on field 'procedure'"):
        validate_model_output(
            _query([{"field": "procedure", "op": "eq", "value": "MRI brain"}]),
            catalog=MOCK_CATALOG,
        )


def test_query_aggregate_rejects_unknown_schema_with_catalog():
    with pytest.raises(ValueError, match="unknown dataset/schema"):
        validate_model_output(_query(MRI_BY_ER, schema="nope"), catalog=MOCK_CATALOG)


def test_structured_output_schema_describes_filter_clauses():
    tool_args = OPENAI_STRUCTURED_OUTPUT_SCHEMA["properties"]["tool_args"]
    assert tool_args["required"] == ["dataset", "schema", "filters"]
    item = tool_args["properties"]["filters"]["items"]
    assert "phrase" in item["properties"]["op"]["enum"]
    assert item["additionalProperties"] is False
