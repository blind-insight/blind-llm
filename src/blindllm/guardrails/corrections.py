"""Deterministic post-processing for multi-query BlindLLM answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_AGG_RE = re.compile(r"^(?P<op>count|avg|sum|min|max)\((?P<arg>.+)\)$", re.I)
_COUNT_ONLY_FIELD = "records"


@dataclass(frozen=True, slots=True)
class ParsedAggregate:
    dataset: str
    schema: str
    field: str
    op: str
    filters: tuple[str, ...]
    value: float


def summarize_computed_aggregates(
    payloads: dict[str, Any],
) -> dict[str, list[dict[str, Any]]] | None:
    """Compute safe summaries from completed aggregate tool payloads.

    This is intentionally conservative: pooled averages require count+avg for
    at least two matching branches, and OR-count totals require branches that
    differ by exactly one equality field with otherwise identical filters.
    """
    aggregates = [
        parsed
        for payload in payloads.values()
        if (parsed := _parse_payload(payload)) is not None
    ]
    weighted_averages = _weighted_averages(aggregates)
    or_counts = _or_count_totals(aggregates)

    summary: dict[str, list[dict[str, Any]]] = {}
    if weighted_averages:
        summary["weighted_averages"] = weighted_averages
    if or_counts:
        summary["or_counts"] = or_counts
    return summary or None


def build_computed_summary_message(summary: dict[str, list[dict[str, Any]]]) -> str:
    """Create the follow-up prompt appended before the next model turn."""
    return (
        "Computed aggregate summary (server-side, deterministic):\n"
        f"```json\n{json.dumps(summary, indent=2, sort_keys=True)}\n```\n\n"
        "Use these computed values when answering average or OR-style questions. "
        "Answer concisely; do not narrate every backend query unless asked."
    )


def build_or_average_correction_message() -> str:
    """Prompt the model to finish OR-average child queries before answering."""
    return (
        "Your final answer arrived before the OR-average could be computed. "
        "Run separate serial query_aggregate calls for each OR branch: one count "
        "and one avg per branch on the same numeric field, then provide "
        "final_answer using the server-computed weighted average."
    )


def summary_signature(summary: dict[str, list[dict[str, Any]]]) -> str:
    """Stable key used to avoid appending duplicate computed summaries."""
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def prompt_requests_average(prompt: str) -> bool:
    lower = prompt.lower()
    return any(term in lower for term in ("average", " avg ", "mean"))


def prompt_requests_or_average(prompt: str) -> bool:
    lower = prompt.lower()
    has_or = " or " in f" {lower} " or " either " in lower
    return prompt_requests_average(prompt) and has_or


def needs_or_average_followup(prompt: str, payloads: dict[str, Any]) -> bool:
    if not prompt_requests_or_average(prompt):
        return False
    summary = summarize_computed_aggregates(payloads)
    if summary and summary.get("weighted_averages"):
        return False
    return True


def needs_avg_combo_followup(prompt: str, payloads: dict[str, Any]) -> bool:
    """True when the prompt asks for an average but no `avg` aggregate ran.

    Blocks sum÷count-derived averages: if the model only queried sum and/or
    count, the deterministic correction tells it to re-query with a single
    combo filter that applies `avg` directly (one proxy query).
    """
    if not prompt_requests_average(prompt):
        return False
    aggregates = [
        parsed
        for payload in payloads.values()
        if (parsed := _parse_payload(payload)) is not None
    ]
    if not aggregates:
        return False
    if any(aggregate.op == "avg" for aggregate in aggregates):
        return False
    return any(aggregate.op in {"sum", "count"} for aggregate in aggregates)


def build_avg_combo_correction_message() -> str:
    """Prompt the model to use a direct avg combo filter, not sum÷count."""
    return (
        "Do not derive an average by dividing a sum by a count. Re-query "
        "with a single combo filter that applies avg directly to the "
        "numeric field, scoped by the other clauses (for example "
        "`year:2021~2025,risk_level:avg(0~100)`), then provide final_answer "
        "using the returned value."
    )


def _parse_payload(payload: Any) -> ParsedAggregate | None:
    if parsed := _parse_aggregate_payload(payload):
        return parsed
    return _parse_count_only_payload(payload)


def _parse_aggregate_payload(payload: Any) -> ParsedAggregate | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("metric_scope") != "aggregate_only":
        return None
    if payload.get("query_mode") != "aggregate":
        return None

    agg_filter = payload.get("filter")
    if not isinstance(agg_filter, str):
        return None
    dataset = payload.get("dataset")
    schema = payload.get("schema")
    if not isinstance(dataset, str) or not isinstance(schema, str):
        return None

    value = payload.get("value")
    if not isinstance(value, (int, float)):
        return None

    clauses = [clause.strip() for clause in agg_filter.split(",") if clause.strip()]
    aggregate_clause: tuple[str, str] | None = None
    filters: list[str] = []

    for clause in clauses:
        if ":" not in clause:
            return None
        field, raw_value = (part.strip() for part in clause.split(":", 1))
        match = _AGG_RE.match(raw_value)
        if match:
            if aggregate_clause is not None:
                return None
            aggregate_clause = (field, match.group("op").lower())
        else:
            filters.append(f"{field}:{raw_value}")

    if aggregate_clause is None:
        return None

    field, op = aggregate_clause
    return ParsedAggregate(
        dataset=dataset,
        schema=schema,
        field=field,
        op=op,
        filters=tuple(sorted(filters)),
        value=float(value),
    )


def _parse_count_only_payload(payload: Any) -> ParsedAggregate | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("metric_scope") != "count_only":
        return None
    if payload.get("query_mode") != "count_only":
        return None

    agg_filter = payload.get("filter")
    if not isinstance(agg_filter, str):
        return None
    dataset = payload.get("dataset")
    schema = payload.get("schema")
    if not isinstance(dataset, str) or not isinstance(schema, str):
        return None

    value = payload.get("value")
    if not isinstance(value, (int, float)):
        return None

    filters: list[str] = []
    for clause in agg_filter.split(","):
        clause = clause.strip()
        if not clause or ":" not in clause:
            return None
        field, raw_value = (part.strip() for part in clause.split(":", 1))
        if _AGG_RE.match(raw_value):
            return None
        filters.append(f"{field}:{raw_value}")

    if not filters:
        return None

    return ParsedAggregate(
        dataset=dataset,
        schema=schema,
        field=_COUNT_ONLY_FIELD,
        op="count",
        filters=tuple(sorted(filters)),
        value=float(value),
    )


def _weighted_averages(aggregates: list[ParsedAggregate]) -> list[dict[str, Any]]:
    by_branch: dict[tuple[str, str, str, tuple[str, ...]], dict[str, float]] = {}
    for aggregate in aggregates:
        if aggregate.op not in {"count", "avg"}:
            continue
        key = (
            aggregate.dataset,
            aggregate.schema,
            aggregate.field,
            aggregate.filters,
        )
        by_branch.setdefault(key, {})[aggregate.op] = aggregate.value

    by_metric: dict[
        tuple[str, str, str], list[tuple[tuple[str, ...], dict[str, float]]]
    ]
    by_metric = {}
    for (dataset, schema, field, filters), values in by_branch.items():
        if {"count", "avg"} <= values.keys():
            by_metric.setdefault((dataset, schema, field), []).append((filters, values))

    summaries: list[dict[str, Any]] = []
    for (dataset, schema, field), branches in sorted(by_metric.items()):
        if len(branches) < 2:
            continue
        branches = sorted(branches, key=lambda branch: branch[0])
        total_count = sum(values["count"] for _, values in branches)
        if total_count <= 0:
            continue
        weighted = (
            sum(values["count"] * values["avg"] for _, values in branches) / total_count
        )
        summaries.append(
            {
                "dataset": dataset,
                "schema": schema,
                "field": field,
                "branches": [_format_branch(filters) for filters, _ in branches],
                "total_count": total_count,
                "weighted_average": weighted,
            }
        )
    return summaries


def _or_count_totals(aggregates: list[ParsedAggregate]) -> list[dict[str, Any]]:
    counts = [aggregate for aggregate in aggregates if aggregate.op == "count"]
    grouped: dict[tuple[str, str, str, tuple[str, ...], str], list[ParsedAggregate]]
    grouped = {}

    for aggregate in counts:
        parsed_filters = [_split_filter(filter_) for filter_ in aggregate.filters]
        for idx, (field, _value) in enumerate(parsed_filters):
            shared = tuple(
                f"{other_field}:{other_value}"
                for j, (other_field, other_value) in enumerate(parsed_filters)
                if j != idx
            )
            key = (
                aggregate.dataset,
                aggregate.schema,
                aggregate.field,
                shared,
                field,
            )
            grouped.setdefault(key, []).append(aggregate)

    summaries: list[dict[str, Any]] = []
    for (dataset, schema, field, shared, or_field), branches in sorted(grouped.items()):
        if len(branches) < 2:
            continue
        branch_rows = sorted(
            (
                {
                    "filter": _format_branch(branch.filters),
                    "or_value": next(
                        value
                        for filter_field, value in map(_split_filter, branch.filters)
                        if filter_field == or_field
                    ),
                    "count": branch.value,
                }
                for branch in branches
            ),
            key=lambda row: row["or_value"],
        )
        or_values = [row["or_value"] for row in branch_rows]
        if len(or_values) != len(set(or_values)):
            continue
        largest = max(branch_rows, key=lambda row: row["count"])
        summaries.append(
            {
                "dataset": dataset,
                "schema": schema,
                "field": field,
                "shared_filters": list(shared),
                "or_field": or_field,
                "or_values": sorted(or_values),
                "branches": branch_rows,
                "total_count": sum(row["count"] for row in branch_rows),
                "largest_branch": largest["or_value"],
                "largest_count": largest["count"],
            }
        )
    return summaries


def _split_filter(filter_: str) -> tuple[str, str]:
    field, value = filter_.split(":", 1)
    return field, value


def _format_branch(filters: tuple[str, ...]) -> str:
    return ",".join(filters) if filters else "<all>"
