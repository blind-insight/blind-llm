"""Server-side aggregate post-processing for multi-query answers."""

from __future__ import annotations

import pytest

from blindllm.guardrails.corrections import (
    needs_sum_count_followup,
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


def test_computes_weighted_average_from_branch_sums_and_counts():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload("id:count(1~999),name:Bob", 10),
            "query_aggregate:1": _payload("age:sum(0~100),name:Bob", 200),
            "query_aggregate:2": _payload("id:count(1~999),name:Ann", 30),
            "query_aggregate:3": _payload("age:sum(0~100),name:Ann", 1200),
        }
    )

    assert summary is not None
    assert [(row["branch"], row["average"]) for row in summary["averages"]] == [
        ("name:Ann", 40.0),
        ("name:Bob", 20.0),
    ]
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


def test_single_branch_average_is_exact_sum_over_count():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload(
                "billed:sum(0~40000),dept:oncology", 18500361
            ),
            "query_aggregate:1": _payload("id:count(1~999),dept:oncology", 1083),
        }
    )

    assert summary is not None
    [row] = summary["averages"]
    # Not the proxy's rounded avg (17082): the cents survive.
    assert round(row["average"], 2) == 17082.51
    assert "weighted_averages" not in summary


def test_count_only_branch_pairs_with_a_sum():
    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": _payload("age:sum(0~100),name:Bob", 200),
            "query_aggregate:1": _count_only_payload("name:Bob", 10),
        }
    )

    assert summary is not None
    assert summary["averages"][0]["average"] == 20.0


@pytest.mark.parametrize(
    "payloads",
    [
        {
            "query_aggregate:0": _payload("age:avg(0~100),name:Bob", 20),
            "query_aggregate:1": _payload("age:avg(0~100),name:Ann", 40),
        },
        {
            "query_aggregate:0": _payload("age:sum(0~100),name:Bob", 200),
            "query_aggregate:1": _payload("id:count(1~999),name:Ann", 30),
        },
    ],
)
def test_no_average_without_a_matching_sum_and_count(payloads):
    assert summarize_computed_aggregates(payloads) is None


# -- needs_sum_count_followup: averages are sum ÷ count, never avg -----------

_AVG_PROMPT = "What is the average risk score for fraud reports between 2021 and 2025?"


def test_followup_flags_avg_queries():
    payloads = {
        "query_aggregate:0": _payload("risk:avg(0~101),year:2021~2025", 47),
    }
    assert needs_sum_count_followup(_AVG_PROMPT, payloads) is True


def test_followup_flags_a_sum_without_its_count():
    payloads = {
        "query_aggregate:0": _payload("risk:sum(0~101),year:2021~2025", 3268),
    }
    assert needs_sum_count_followup(_AVG_PROMPT, payloads) is True


def test_followup_satisfied_by_matching_sum_and_count():
    payloads = {
        "query_aggregate:0": _payload("risk:sum(0~101),year:2021~2025", 3268),
        "query_aggregate:1": _payload("id:count(1~999),year:2021~2025", 830),
    }
    assert needs_sum_count_followup(_AVG_PROMPT, payloads) is False


def test_followup_ignores_non_average_prompts():
    payloads = {
        "query_aggregate:0": _payload("risk:avg(0~101),year:2021~2025", 47),
    }
    assert needs_sum_count_followup("How many fraud reports?", payloads) is False


def test_followup_inert_before_any_aggregate_query():
    assert needs_sum_count_followup(_AVG_PROMPT, {}) is False


def _structured_payload(filters: list[dict], value: float) -> dict:
    return {
        "metric_scope": "aggregate_only",
        "query_mode": "aggregate",
        "dataset": "demo",
        "schema": "visits",
        "filters": filters,
        "value": value,
        "contains_pii": False,
    }


def test_weighted_average_from_structured_phrase_branches():
    def branch(phrase: str, field: str, op: str, value: float) -> dict:
        return _structured_payload(
            [
                {"field": "note", "op": "phrase", "value": phrase},
                {"field": field, "op": op, "value": "0~40000"},
            ],
            value,
        )

    summary = summarize_computed_aggregates(
        {
            "query_aggregate:0": branch("chest pain, radiating", "id", "count", 10),
            "query_aggregate:1": branch("chest pain, radiating", "billed", "sum", 1000),
            "query_aggregate:2": branch("brain fog", "id", "count", 30),
            "query_aggregate:3": branch("brain fog", "billed", "sum", 6000),
        }
    )

    assert summary is not None
    [weighted] = summary["weighted_averages"]
    # The comma inside the phrase stays one clause instead of splitting.
    assert weighted["branches"] == ['note:"brain fog"', 'note:"chest pain, radiating"']
    assert weighted["weighted_average"] == 175.0


def test_structured_payload_with_invalid_filters_is_ignored():
    payload = _structured_payload([{"field": "note", "op": "phrase", "value": "!!"}], 3)
    assert summarize_computed_aggregates({"q": payload}) is None
