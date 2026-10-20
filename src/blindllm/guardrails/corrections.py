"""Deterministic post-processing for multi-query BlindLLM answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from blindllm.filters import PHRASE, parse_filters

_AGG_RE = re.compile(r"^(?P<op>count|avg|sum|min|max)\((?P<arg>.+)\)$", re.IGNORECASE)
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

    Averages are always sum ÷ count, never the proxy's rounded ``avg``: each
    branch (one set of non-aggregate filters) with both a ``sum`` and a
    ``count`` gets an exact average, and two or more such branches on the same
    field pool into a weighted average, Σsum ÷ Σcount. OR-count totals require
    branches that differ by exactly one equality field with otherwise
    identical filters.

    Pairing assumes the sum and the count each run over their field's full
    declared range, as the system prompt requires; a narrower bound is itself
    a filter and would make the pair disagree.
    """
    aggregates = [
        parsed
        for payload in payloads.values()
        if (parsed := _parse_payload(payload)) is not None
    ]
    averages = _averages(aggregates)
    weighted_averages = _weighted_averages(averages)
    or_counts = _or_count_totals(aggregates)

    summary: dict[str, list[dict[str, Any]]] = {}
    if averages:
        summary["averages"] = averages
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
        "Run separate serial query_aggregate calls for each OR branch: one sum "
        "of the numeric field and one count per branch, with identical "
        "clauses, then provide final_answer using the server-computed "
        "weighted average."
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
    return not (summary and summary.get("weighted_averages"))


def needs_sum_count_followup(prompt: str, payloads: dict[str, Any]) -> bool:
    """True when an average is being built from ``avg`` or from an unpaired sum.

    The proxy's ``avg`` rounds to a whole number, so averages must be sum ÷
    count. This flags a run that used ``avg``, or that queried a ``sum`` with
    no ``count`` carrying the same clauses, so the correction can send the
    model back for the missing query.
    """
    if not prompt_requests_average(prompt):
        return False
    aggregates = [
        parsed
        for payload in payloads.values()
        if (parsed := _parse_payload(payload)) is not None
    ]
    if any(aggregate.op == "avg" for aggregate in aggregates):
        return True
    counted = {(a.dataset, a.schema, a.filters) for a in aggregates if a.op == "count"}
    return any(
        a.op == "sum" and (a.dataset, a.schema, a.filters) not in counted
        for a in aggregates
    )


def build_sum_count_correction_message() -> str:
    """Prompt the model to compute an average as sum ÷ count, never avg."""
    return (
        "Do not use avg: the proxy rounds it to a whole number. For each "
        "average, run two query_aggregate calls with identical clauses: a sum "
        "of the numeric field over its full declared range, and a count on "
        "the schema's count_vehicle over its full range. Then provide "
        "final_answer using sum ÷ count."
    )


def _parse_payload(payload: Any) -> ParsedAggregate | None:
    if parsed := _parse_aggregate_payload(payload):
        return parsed
    return _parse_count_only_payload(payload)


def _payload_clauses(payload: dict[str, Any]) -> list[tuple[str, str]] | None:
    """``(field, value)`` pairs from a payload's structured ``filters`` or legacy ``filter``.

    Aggregates render as ``op(bounds)`` so ``_AGG_RE`` recognizes them, and
    phrases are quoted so a phrase never collides with an equal-valued eq clause.
    """
    structured = payload.get("filters")
    if isinstance(structured, list):
        try:
            parsed = parse_filters(structured)
        except ValueError:
            return None
        return [
            (c.field, f'"{c.value}"' if c.op == PHRASE else c.to_wire()["value"])
            for c in parsed
        ]

    legacy = payload.get("filter")
    if not isinstance(legacy, str):
        return None
    clauses: list[tuple[str, str]] = []
    for clause in legacy.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if ":" not in clause:
            return None
        field, raw_value = (part.strip() for part in clause.split(":", 1))
        clauses.append((field, raw_value))
    return clauses


def _parse_aggregate_payload(payload: Any) -> ParsedAggregate | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("metric_scope") != "aggregate_only":
        return None
    if payload.get("query_mode") != "aggregate":
        return None

    clauses = _payload_clauses(payload)
    if clauses is None:
        return None
    dataset = payload.get("dataset")
    schema = payload.get("schema")
    if not isinstance(dataset, str) or not isinstance(schema, str):
        return None

    value = payload.get("value")
    if not isinstance(value, (int, float)):
        return None

    aggregate_clause: tuple[str, str] | None = None
    filters: list[str] = []

    for field, raw_value in clauses:
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

    clauses = _payload_clauses(payload)
    if clauses is None:
        return None
    dataset = payload.get("dataset")
    schema = payload.get("schema")
    if not isinstance(dataset, str) or not isinstance(schema, str):
        return None

    value = payload.get("value")
    if not isinstance(value, (int, float)):
        return None

    filters: list[str] = []
    for field, raw_value in clauses:
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


def _averages(aggregates: list[ParsedAggregate]) -> list[dict[str, Any]]:
    """Exact sum ÷ count per branch, for every field that has both."""
    counts: dict[tuple[str, str, tuple[str, ...]], float] = {}
    sums: dict[tuple[str, str, tuple[str, ...]], dict[str, float]] = {}
    for aggregate in aggregates:
        branch = (aggregate.dataset, aggregate.schema, aggregate.filters)
        if aggregate.op == "count":
            counts[branch] = aggregate.value
        elif aggregate.op == "sum":
            sums.setdefault(branch, {})[aggregate.field] = aggregate.value

    rows: list[dict[str, Any]] = []
    for branch in sorted(sums.keys() & counts.keys()):
        dataset, schema, filters = branch
        count = counts[branch]
        if count <= 0:
            continue
        for field, total in sorted(sums[branch].items()):
            rows.append(
                {
                    "dataset": dataset,
                    "schema": schema,
                    "field": field,
                    "branch": _format_branch(filters),
                    "sum": total,
                    "count": count,
                    "average": total / count,
                }
            )
    return rows


def _weighted_averages(averages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool two or more branches of the same field: Σsum ÷ Σcount."""
    by_metric: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in averages:
        key = (row["dataset"], row["schema"], row["field"])
        by_metric.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, schema, field), branches in sorted(by_metric.items()):
        if len(branches) < 2:
            continue
        total_count = sum(row["count"] for row in branches)
        summaries.append(
            {
                "dataset": dataset,
                "schema": schema,
                "field": field,
                "branches": [row["branch"] for row in branches],
                "total_count": total_count,
                "weighted_average": sum(row["sum"] for row in branches) / total_count,
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
