"""Unit tests for the shared `prompts.py` helpers.

Exercised indirectly via orchestrator tests, but these direct cases lock
in the message shape that all three providers depend on.
"""

from __future__ import annotations

from blindllm.prompts import (
    SYSTEM_PROMPT,
    append_tool_result,
    build_initial_messages,
    catalog_summary,
)


class TestCatalogSummary:
    def test_empty_catalog_returns_fallback_guidance(self):
        out = catalog_summary([])
        assert "empty" in out.lower()
        assert "list_schemas" in out

    def test_renders_dataset_schema_label_description(self):
        out = catalog_summary(
            [
                {
                    "dataset": "fraud-data",
                    "schema": "train",
                    "label": "Fraud account records",
                    "description": "Account-level signals.",
                }
            ]
        )
        assert "fraud-data/train" in out
        assert "Fraud account records" in out
        assert "Account-level signals." in out

    def test_falls_back_to_dataset_slash_schema_when_label_missing(self):
        out = catalog_summary([{"dataset": "bc", "schema": "v1", "description": "x"}])
        # Label defaults to "<dataset>/<schema>" when not provided.
        assert "bc/v1" in out

    def test_handles_empty_description(self):
        out = catalog_summary(
            [
                {
                    "dataset": "d",
                    "schema": "s",
                    "label": "label",
                    "description": "",
                }
            ]
        )
        assert "d/s" in out
        # Trailing whitespace from the empty description is stripped.
        assert "d/s — label." in out


class TestBuildInitialMessages:
    def _user_message(self, messages):
        return next(m for m in messages if m["role"] == "user")

    def test_system_prompt_is_first(self):
        msgs = build_initial_messages("hi", {}, [], [])
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_system_prompt_avoids_human_impersonation_language(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "human analyst" not in prompt
        assert "ai assistant" in prompt
        assert "do not claim to be human" in prompt

    def test_current_date_is_second(self):
        msgs = build_initial_messages("hi", {}, [], [])
        assert msgs[1]["role"] == "system"
        assert "Current date:" in msgs[1]["content"]

    def test_catalog_summary_is_third(self):
        msgs = build_initial_messages("hi", {}, [], [])
        assert msgs[2]["role"] == "system"
        assert "list_schemas" in msgs[2]["content"]

    def test_current_date_context_formats_utc_date(self):
        from datetime import datetime, timezone

        from blindllm.prompts import current_date_context

        fixed = datetime(2026, 7, 14, 3, 2, 1, tzinfo=timezone.utc)
        assert "Current date: 2026-07-14 (UTC)" in current_date_context(fixed)

    def test_time_ranges_document_relative_windows(self):
        assert "the past 3 months" in SYSTEM_PROMPT
        assert "range `202605~202607`" in SYSTEM_PROMPT

    def test_user_message_is_last(self):
        msgs = build_initial_messages("the prompt", {}, [], [])
        assert msgs[-1] == {"role": "user", "content": "the prompt"}

    def test_hints_become_a_system_message_when_present(self):
        msgs = build_initial_messages("hi", {}, [], ["fraud-data"])
        # Match the canonical hint marker, not any prose that mentions
        # `@-mention` (the system prompt itself documents that override).
        hint_msg = [m for m in msgs if "User hinted at datasets" in m["content"]]
        assert len(hint_msg) == 1
        assert "fraud-data" in hint_msg[0]["content"]

    def test_no_hint_message_when_empty(self):
        msgs = build_initial_messages("hi", {}, [], [])
        assert not any("User hinted at datasets" in m["content"] for m in msgs)

    def test_context_is_serialized_as_system_message(self):
        msgs = build_initial_messages("hi", {"hinted_datasets": ["fraud-data"]}, [], [])
        ctx = [m for m in msgs if "Additional context" in m["content"]]
        assert len(ctx) == 1
        assert "fraud-data" in ctx[0]["content"]

    def test_empty_context_produces_no_context_message(self):
        msgs = build_initial_messages("hi", {}, [], [])
        assert not any("Additional context" in m["content"] for m in msgs)


class TestAppendToolResult:
    def test_appends_assistant_then_user_messages(self):
        messages: list[dict] = []
        out = append_tool_result(
            messages,
            assistant_output={"response_type": "tool_call"},
            tool_name="list_schemas",
            tool_args={"dataset": None, "schema": None, "filters": None},
            tool_result={"schemas": []},
        )
        assert len(out) == 2
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"

    def test_user_message_references_the_tool_name(self):
        messages: list[dict] = []
        append_tool_result(
            messages,
            assistant_output={},
            tool_name="query_aggregate",
            tool_args={
                "dataset": "x",
                "schema": "y",
                "filters": [{"field": "z", "op": "eq", "value": "1"}],
            },
            tool_result={"value": 5},
        )
        assert "query_aggregate" in messages[1]["content"]
        assert '"field": "z"' in messages[1]["content"]
        assert "5" in messages[1]["content"]

    def test_mutates_in_place_and_returns_same_list(self):
        messages: list[dict] = []
        out = append_tool_result(
            messages,
            assistant_output={},
            tool_name="t",
            tool_args={},
            tool_result={},
        )
        assert out is messages


class TestLanguageNormalization:
    """SYSTEM_PROMPT must guide providers to normalize user phrasing before tool calls."""

    def test_prompt_includes_language_normalization_section(self):
        assert "LANGUAGE NORMALIZATION" in SYSTEM_PROMPT

    def test_time_range_synonyms_map_to_year_range_clause(self):
        prompt = SYSTEM_PROMPT.lower()
        assert '{"field":"year","op":"range","value":"2022~2024"}' in prompt
        assert "from 2022 to 2024" in prompt
        assert "between 2022 and 2024" in prompt
        assert "through 2024" in prompt
        assert "thru 2024" in prompt

    def test_average_synonyms_map_to_sum_over_count(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "mean" in prompt
        assert "typical" in prompt
        assert "then sum ÷ count. never `avg`: it rounds." in prompt

    def test_count_synonyms_map_to_count_semantics(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "total number" in prompt
        assert "how many" in prompt

    def test_combo_filter_example_includes_year_range_and_aggregate(self):
        assert (
            "range year 2022~2024 + eq account_jurisdiction US + max risk_level 0~100"
            in SYSTEM_PROMPT
        )
        assert "An average in a combo question is still a sum query" in SYSTEM_PROMPT
        assert (
            "never invent `train`" in SYSTEM_PROMPT.lower()
            or "fraud-train" in SYSTEM_PROMPT
        )

    def test_fuzzy_language_does_not_invent_operators(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "approximately" in prompt
        assert "do not" in prompt and "invent" in prompt
        assert "fuzzy" in prompt

    def test_or_language_keeps_separate_serial_queries(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "separate serial" in prompt
        assert "never encode or inside one clause" in prompt

    def test_follow_up_drill_down_keeps_prior_filters(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "follow-up / drill-down scope" in prompt
        assert "of those" in prompt
        assert "eq account_jurisdiction fr" in prompt
        assert "must be <=" in prompt


class TestPhraseSearch:
    """SYSTEM_PROMPT must teach phrase clauses on `text: true` fields."""

    def test_documents_phrase_op_and_text_fields(self):
        assert "FREE-TEXT / PHRASE SEARCH" in SYSTEM_PROMPT
        assert '"op":"phrase"' in SYSTEM_PROMPT
        assert "`text: true`" in SYSTEM_PROMPT

    def test_documents_matching_semantics(self):
        prompt = SYSTEM_PROMPT.lower()
        assert "consecutively and in order" in prompt
        assert "no stemming" in prompt
        assert "matches negated mentions" in prompt

    def test_teaches_record_wording_over_user_wording(self):
        assert "phrase `MRI` on the procedure field" in SYSTEM_PROMPT

    def test_phrase_counts_use_the_count_vehicle(self):
        assert "Phrase-scoped counts" in SYSTEM_PROMPT

    def test_citations_carry_a_plain_language_query_line(self):
        assert 'procedure contains "MRI"' in SYSTEM_PROMPT


class TestAverages:
    """Averages and rates are sum ÷ count; the proxy's avg rounds."""

    def test_prompt_forbids_avg_and_explains_why(self):
        assert "AVERAGES AND RATES — always sum ÷ count, never `avg`" in SYSTEM_PROMPT
        assert "rounds it to a whole" in SYSTEM_PROMPT

    def test_average_example_is_a_sum_and_a_count_with_identical_clauses(self):
        assert '{"field":"billed_amount","op":"sum","value":"0~39973"}' in SYSTEM_PROMPT
        assert "IDENTICAL eq / range / phrase clauses" in SYSTEM_PROMPT

    def test_rates_of_flags_use_the_same_pattern(self):
        assert "sum of `readmitted_30d` (range 0~1) ÷ count" in SYSTEM_PROMPT

    def test_no_example_uses_the_avg_op(self):
        assert '"op":"avg"' not in SYSTEM_PROMPT

    def test_query_budget_leaves_room_for_two_queries_per_average(self):
        assert "≤ 20" in SYSTEM_PROMPT
