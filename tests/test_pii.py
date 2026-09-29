"""Leakage policy unit tests."""

from __future__ import annotations

from blindllm.guardrails.pii import (
    PII_PATTERN,
    validate_narrative,
    validate_tool_payload,
)


class TestValidateToolPayload:
    def test_aggregate_only_payload_is_allowed(self):
        decision = validate_tool_payload(
            {
                "metric_scope": "aggregate_only",
                "total_entities": 100,
                "flagged_entities": 5,
                "contains_pii": False,
            }
        )
        assert decision.allowed is True

    def test_raw_records_are_blocked(self):
        decision = validate_tool_payload({"raw_records": [{"id": 1}]})
        assert decision.allowed is False
        assert "raw_records" in decision.reason

    def test_identifiers_are_blocked(self):
        decision = validate_tool_payload({"identifiers": ["a@example.com"]})
        assert decision.allowed is False
        assert "identifiers" in decision.reason

    def test_contains_pii_flag_blocks(self):
        decision = validate_tool_payload({"contains_pii": True, "count": 5})
        assert decision.allowed is False
        assert "PII" in decision.reason

    def test_pii_token_in_string_value_blocks(self):
        decision = validate_tool_payload({"note": "Email: jane@example.com is suspect"})
        assert decision.allowed is False
        assert "note" in decision.reason


class TestValidateNarrative:
    def test_clean_narrative_is_allowed(self):
        decision = validate_narrative(
            "The flagged ratio in region US rose to 9% in Q4 — a 2% drift."
        )
        assert decision.allowed is True

    def test_ssn_format_blocks(self):
        decision = validate_narrative("Subject SSN 123-45-6789 was flagged.")
        assert decision.allowed is False

    def test_email_blocks(self):
        decision = validate_narrative("Contact admin@example.com for details.")
        assert decision.allowed is False

    def test_long_digit_string_blocks(self):
        # 9-digit numeric block (e.g., raw SSN)
        decision = validate_narrative("Subject ID 123456789 was flagged.")
        assert decision.allowed is False


def test_pii_pattern_compiled():
    assert PII_PATTERN.search("jane@example.com") is not None
    assert PII_PATTERN.search("the ratio is 0.42") is None
