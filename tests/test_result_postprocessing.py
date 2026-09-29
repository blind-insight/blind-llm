"""Server-side aggregate post-processing for multi-query answers."""

from __future__ import annotations

import pytest

from blindllm.guardrails.corrections import (
    needs_avg_combo_followup,
    summarize_computed_aggregates,
)


def _payload(filter_: str, value: float) -> dict:
    return {
        "metric_scope": "aggregate_only",
        "query_mode": "aggregate",
        "dataset": "demo",
        "schema": "people",
        "filter": filter_,
        "value": value,
        "contains_pii": False,
    }


def test_computes_weighted_average_from_branch_counts_and_averages():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload("age:count(0~100),name:Bob", 10),
            "query_aggregate:1": _payload("age:avg(0~100),name:Bob", 20),
            "query_aggregate:2": _payload("age:count(0~100),name:Ann", 30),
            "query_aggregate:3": _payload("age:avg(0~100),name:Ann", 40),
        }
    )

    assert summary is not None
    assert summary["weighted_averages"] == [
        {
            "dataset": "demo",
            "schema": "people",
            "field": "age",
            "branches": ["name:Ann", "name:Bob"],
            "total_count": 40.0,
            "weighted_average": 35.0,
        }
    ]


def _count_only_payload(filter_: str, value: float) -> dict:
    return {
        "metric_scope": "count_only",
        "query_mode": "count_only",
        "dataset": "demo",
        "schema": "people",
        "filter": filter_,
        "value": value,
        "contains_pii": False,
    }


def test_computes_or_count_total_for_count_only_name_branches():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _count_only_payload("name:Bob", 10),
            "query_aggregate:1": _count_only_payload("name:Angie", 30),
        }
    )

    assert summary is not None
    assert summary["or_counts"] == [
        {
            "dataset": "demo",
            "schema": "people",
            "field": "records",
            "shared_filters": [],
            "or_field": "name",
            "or_values": ["Angie", "Bob"],
            "branches": [
                {"filter": "name:Angie", "or_value": "Angie", "count": 30.0},
                {"filter": "name:Bob", "or_value": "Bob", "count": 10.0},
            ],
            "total_count": 40.0,
            "largest_branch": "Angie",
            "largest_count": 30.0,
        }
    ]


def test_computes_or_count_total_for_mutually_exclusive_branch_values():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload("risk:count(0~100),kind:a,year:2025", 4),
            "query_aggregate:1": _payload("risk:count(0~100),kind:b,year:2025", 6),
        }
    )

    assert summary is not None
    assert summary["or_counts"] == [
        {
            "dataset": "demo",
            "schema": "people",
            "field": "risk",
            "shared_filters": ["year:2025"],
            "or_field": "kind",
            "or_values": ["a", "b"],
            "branches": [
                {
                    "filter": "kind:a,year:2025",
                    "or_value": "a",
                    "count": 4.0,
                },
                {
                    "filter": "kind:b,year:2025",
                    "or_value": "b",
                    "count": 6.0,
                },
            ],
            "total_count": 10.0,
            "largest_branch": "b",
            "largest_count": 6.0,
        }
    ]


def test_skips_or_count_total_when_branches_can_overlap():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload("risk:count(0~100),kind:a", 4),
            "query_aggregate:1": _payload("risk:count(0~100),year:2025", 6),
        }
    )

    assert summary is None


@pytest.mark.parametrize(
    "payloads",
    [
        {
            "query_aggregate:0": _payload("age:avg(0~100),name:Bob", 20),
            "query_aggregate:1": _payload("age:avg(0~100),name:Ann", 40),
        },
        {
            "query_aggregate:0": _payload("age:count(0~100),name:Bob", 10),
            "query_aggregate:1": _payload("age:avg(0~100),name:Bob", 20),
        },
    ],
)
def test_skips_weighted_average_without_multiple_complete_branches(payloads):
    assert summarize_computed_aggregates(payloads) is None


# -- needs_avg_combo_followup: block sum÷count-derived averages ---------------

_AVG_PROMPT = "What is the average risk score for fraud reports between 2021 and 2025?"


def test_avg_combo_followup_blocks_sum_count_division():
    payloads = {
        "query_aggregate:0": _payload("risk:count(0~101),year:2021~2025", 830),
        "query_aggregate:1": _payload("risk:sum(0~101),year:2021~2025", 3268),
    }
    assert needs_avg_combo_followup(_AVG_PROMPT, payloads) is True


def test_avg_combo_followup_satisfied_once_avg_query_ran():
    payloads = {
        "query_aggregate:0": _payload("risk:avg(0~101),year:2021~2025", 47.3),
    }
    assert needs_avg_combo_followup(_AVG_PROMPT, payloads) is False


def test_avg_combo_followup_ignores_non_average_prompts():
    payloads = {
        "query_aggregate:0": _payload("risk:count(0~101),year:2021~2025", 830),
    }
    assert needs_avg_combo_followup("How many fraud reports?", payloads) is False


def test_avg_combo_followup_inert_before_any_aggregate_query():
    assert needs_avg_combo_followup(_AVG_PROMPT, {}) is False
