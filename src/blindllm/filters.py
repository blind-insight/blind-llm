"""Structured query filters: one clause per ``{field, op, value}`` object.

The model used to send a filter as one comma-joined string
(``"year:2024,risk_level:avg(0~100)"``). That cannot carry free text: a phrase
like ``chest pain, radiating`` contains the separator, and splitting it yields
two bogus clauses that answer a different question with no error. So
``query_aggregate`` now takes ``filters``, a list of clauses:

    {"field": "procedure",  "op": "phrase", "value": "MRI"}
    {"field": "department", "op": "eq",     "value": "emergency"}
    {"field": "year",       "op": "range",  "value": "2025~2025"}
    {"field": "patient_id", "op": "count",  "value": "100001~139385"}

Clauses are AND-ed. At most one may be an aggregate (``count``/``avg``/``sum``/
``min``/``max``): the server rejects a second one.

``phrase`` matches a consecutive token sequence inside a ``text: true`` field.
The tokenizer here mirrors the proxy's (``proxy/pkg/text/tokenize.go``):
NFKC-normalize, lowercase, split on anything outside ``[a-z0-9]``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

EQ = "eq"
PHRASE = "phrase"
RANGE = "range"
AGGREGATE_OPS: frozenset[str] = frozenset({"count", "avg", "sum", "min", "max"})
FILTER_OPS: frozenset[str] = frozenset({EQ, PHRASE, RANGE}) | AGGREGATE_OPS

# ``2022~2024`` for a range clause; aggregates also accept one-sided bounds.
_RANGE_RE = re.compile(r"^-?\d+~-?\d+$")
_BOUND_RE = re.compile(r"^(-?\d+~-?\d+|(<=|>=|<|>)-?\d+)$")
_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_NUMERIC_TYPES = {"int", "integer", "number", "float"}


class FilterValidationError(ValueError):
    """A filter clause is malformed or not allowed on its field."""


class PhraseValidationError(FilterValidationError):
    """A phrase normalizes to zero tokens and must never reach the proxy.

    The proxy does not reject such a phrase: it silently drops the filter, so
    the query carries no constraint and returns every record the caller may
    read, across all schemas (BAC-537). The count path reports 0 for the same
    query, so the leak is easy to miss.
    """


def tokenize(text: str) -> list[str]:
    """NFKC-normalize, lowercase, and split on runs outside ``[a-z0-9]``."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return [token for token in _SEPARATOR_RE.split(folded) if token]


def validate_phrase(value: str, *, field: str = "") -> list[str]:
    """Return the tokens ``value`` matches on, or raise if it has none."""
    tokens = tokenize(value)
    if not tokens:
        where = f" for field {field!r}" if field else ""
        raise PhraseValidationError(
            f"phrase {value!r}{where} normalizes to zero tokens; only [a-z0-9] "
            "runs survive tokenization, and an empty phrase makes the proxy drop "
            "the filter and return unscoped records"
        )
    return tokens


@dataclass(frozen=True, slots=True)
class FilterClause:
    field: str
    op: str
    value: str

    @property
    def is_aggregate(self) -> bool:
        return self.op in AGGREGATE_OPS

    def to_wire(self) -> dict[str, str]:
        """The proxy's ``/api/records/search/`` filter shape: ``{label, value}``.

        Phrase and equality both travel as the plaintext value; the proxy picks
        token matching or whole-value hashing from the field's schema type.
        """
        if self.is_aggregate:
            return {"label": self.field, "value": f"{self.op}({self.value})"}
        return {"label": self.field, "value": self.value}

    def to_legacy(self) -> str:
        """One ``field:value`` clause of the old comma-joined filter string."""
        return f"{self.field}:{self.to_wire()['value']}"

    def explain(self) -> str:
        """A plain-language rendering, for showing the user what was asked."""
        if self.op == PHRASE:
            return f'{self.field} contains "{self.value}"'
        if self.op == EQ:
            return f"{self.field} = {self.value}"
        if self.op == RANGE:
            lo, hi = self.value.split("~")
            return f"{self.field} {lo}–{hi}" if lo != hi else f"{self.field} = {lo}"
        return f"{self.op} of {self.field} over {self.value}"


def parse_filters(raw: Any) -> list[FilterClause]:
    """Validate the model's ``filters`` list and return typed clauses.

    Checks shape only; :func:`check_filters_against_schema` checks the clauses
    against a schema's field descriptors.
    """
    if not isinstance(raw, list) or not raw:
        raise FilterValidationError("filters must be a non-empty list of clauses")

    clauses: list[FilterClause] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise FilterValidationError(f"filters[{idx}] must be an object")
        extra = set(item) - {"field", "op", "value"}
        if extra:
            raise FilterValidationError(
                f"filters[{idx}] has unexpected keys: {sorted(extra)}"
            )
        field, op, value = item.get("field"), item.get("op"), item.get("value")
        if not isinstance(field, str) or not field.strip():
            raise FilterValidationError(
                f"filters[{idx}].field must be a non-empty string"
            )
        if op not in FILTER_OPS:
            raise FilterValidationError(
                f"filters[{idx}].op {op!r} is not one of {sorted(FILTER_OPS)}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise FilterValidationError(
                f"filters[{idx}].value must be a non-empty string"
            )
        value = value.strip()

        if op == PHRASE:
            validate_phrase(value, field=field)
        elif op == RANGE and not _RANGE_RE.match(value):
            raise FilterValidationError(
                f"filters[{idx}]: range value {value!r} must look like 'lo~hi'"
            )
        elif op in AGGREGATE_OPS and not _BOUND_RE.match(value):
            raise FilterValidationError(
                f"filters[{idx}]: {op} value {value!r} must be 'lo~hi' or a bound like '>40'"
            )
        clauses.append(FilterClause(field.strip(), op, value))

    aggregates = [clause for clause in clauses if clause.is_aggregate]
    if len(aggregates) > 1:
        raise FilterValidationError(
            "only one aggregate clause (count/avg/sum/min/max) is allowed per query; "
            f"got {[f'{c.field}:{c.op}' for c in aggregates]}"
        )
    return clauses


def field_ops(field: dict[str, Any]) -> set[str]:
    """Operations a schema field supports, from its descriptor.

    An explicit ``query_ops`` list wins. Otherwise: ``text: true`` string fields
    take ``phrase`` only (the proxy always token-matches them); plain string
    fields take ``eq``; numeric fields take ``eq``, ``range`` and aggregates.
    """
    explicit = field.get("query_ops")
    if isinstance(explicit, list):
        return {"eq" if op == "count_only" else op for op in explicit}
    if field.get("text"):
        return {PHRASE}
    if str(field.get("type", "")).lower() in _NUMERIC_TYPES:
        return {EQ, RANGE} | AGGREGATE_OPS
    return {EQ}


def check_filters_against_schema(
    clauses: list[FilterClause], fields: list[dict[str, Any]]
) -> None:
    """Raise if a clause names an unknown field, an unsupported op, or a bad value."""
    by_name = {field["name"]: field for field in fields if "name" in field}
    for clause in clauses:
        field = by_name.get(clause.field)
        if field is None:
            raise FilterValidationError(
                f"field {clause.field!r} is not in the schema; known fields: {sorted(by_name)}"
            )
        allowed = field_ops(field)
        if clause.op not in allowed:
            raise FilterValidationError(
                f"op {clause.op!r} is not supported on field {clause.field!r}; "
                f"use one of {sorted(allowed)}"
            )
        values = field.get("values")
        if clause.op == EQ and isinstance(values, list) and clause.value not in values:
            raise FilterValidationError(
                f"{clause.field}={clause.value!r} is not a known value; equality is "
                f"exact and case-sensitive, use one of {values}"
            )


def filters_to_wire(clauses: list[FilterClause]) -> list[dict[str, str]]:
    """The ``filters`` array for the proxy's ``/api/records/search/``."""
    return [clause.to_wire() for clause in clauses]


def filters_to_legacy_string(clauses: list[FilterClause]) -> str:
    """The old comma-joined filter string, for executors that still expect it.

    Raises for any value containing a comma, which the old format cannot carry.
    """
    for clause in clauses:
        if "," in clause.value:
            raise FilterValidationError(
                f"{clause.field} value {clause.value!r} contains a comma, which the "
                "legacy filter string cannot represent; send structured filters"
            )
    return ",".join(clause.to_legacy() for clause in clauses)


def explain_filters(clauses: list[FilterClause]) -> str:
    """E.g. ``count of patient_id over 100001~139385 where procedure contains "MRI"``."""
    aggregates = [clause.explain() for clause in clauses if clause.is_aggregate]
    conditions = [clause.explain() for clause in clauses if not clause.is_aggregate]
    head = aggregates[0] if aggregates else "matching records"
    return f"{head} where {' AND '.join(conditions)}" if conditions else head
