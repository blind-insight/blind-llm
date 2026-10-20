"""Structured filter clauses, phrase tokenization, and schema checks."""

from __future__ import annotations

import pytest

from blindllm.catalog import MOCK_CATALOG
from blindllm.filters import (
    FilterValidationError,
    PhraseValidationError,
    check_filters_against_schema,
    explain_filters,
    field_ops,
    filters_to_legacy_string,
    filters_to_wire,
    parse_filters,
    tokenize,
)

EHR_FIELDS = next(e for e in MOCK_CATALOG if e["schema"] == "ehr-data-v3")["fields"]
FRAUD_FIELDS = next(e for e in MOCK_CATALOG if e["dataset"] == "fraud-data")["fields"]


def clause(field: str, op: str, value: str) -> dict:
    return {"field": field, "op": op, "value": value}


class TestTokenize:
    """Parity cases measured against the live proxy (EHRDemo phrase-search handoff)."""

    @pytest.mark.parametrize(
        ("text", "tokens"),
        [
            ("Urinary Tract Infection", ["urinary", "tract", "infection"]),
            ("machine-learning", ["machine", "learning"]),
            ("ICD10 code A10", ["icd10", "code", "a10"]),
            ("rule out MI", ["rule", "out", "mi"]),
            ("café", ["caf"]),
            ("ＭＲＩ", ["mri"]),  # full-width folds under NFKC
            ("???", []),
        ],
    )
    def test_matches_proxy_tokenizer(self, text, tokens):
        assert tokenize(text) == tokens


class TestParseFilters:
    def test_parses_each_op(self):
        clauses = parse_filters(
            [
                clause("note", "phrase", "chest pain"),
                clause("department", "eq", "emergency"),
                clause("year", "range", "2022~2024"),
                clause("risk", "max", "<45"),
            ]
        )
        assert [c.op for c in clauses] == ["phrase", "eq", "range", "max"]

    def test_numeric_values_are_coerced_to_strings(self):
        [c] = parse_filters([{"field": "year", "op": "eq", "value": 2025}])
        assert c.value == "2025"

    @pytest.mark.parametrize("value", ["???", "  ", "中文", "—"])
    def test_rejects_phrases_with_no_tokens(self, value):
        with pytest.raises((PhraseValidationError, FilterValidationError)):
            parse_filters([clause("note", "phrase", value)])

    def test_rejects_second_aggregate(self):
        with pytest.raises(FilterValidationError, match="only one aggregate"):
            parse_filters([clause("a", "count", "0~10"), clause("b", "avg", "0~10")])

    @pytest.mark.parametrize(
        "bad",
        [
            [],
            "year:2024",
            [clause("year", "between", "1~2")],
            [clause("year", "range", "2022-2024")],
            [clause("risk", "avg", "avg(0~100)")],
            [{"field": "a", "op": "eq", "value": "b", "extra": 1}],
            [clause("", "eq", "x")],
        ],
    )
    def test_rejects_malformed_input(self, bad):
        with pytest.raises(FilterValidationError):
            parse_filters(bad)


class TestSchemaChecks:
    def test_field_ops_from_descriptor(self):
        by_name = {f["name"]: f for f in EHR_FIELDS}
        assert field_ops(by_name["note"]) == {"phrase"}
        assert field_ops(by_name["department"]) == {"eq"}
        assert {"range", "count", "avg"} <= field_ops(by_name["billed_amount"])
        assert field_ops({"name": "x", "query_ops": ["count_only"]}) == {"eq"}

    def test_accepts_phrase_on_text_field(self):
        check_filters_against_schema(
            parse_filters(
                [
                    clause("note", "phrase", "brain fog"),
                    clause("age", "range", "18~29"),
                    clause("patient_id", "count", "100001~139385"),
                ]
            ),
            EHR_FIELDS,
        )

    def test_rejects_phrase_on_plain_string(self):
        with pytest.raises(FilterValidationError, match="not supported"):
            check_filters_against_schema(
                parse_filters([clause("department", "phrase", "emerg")]), EHR_FIELDS
            )

    def test_rejects_eq_value_outside_known_values(self):
        with pytest.raises(FilterValidationError, match="use one of"):
            check_filters_against_schema(
                parse_filters([clause("account_jurisdiction", "eq", "Germany")]),
                FRAUD_FIELDS,
            )

    def test_rejects_unknown_field(self):
        with pytest.raises(FilterValidationError, match="not in the schema"):
            check_filters_against_schema(
                parse_filters([clause("symptom", "phrase", "cough")]), EHR_FIELDS
            )


MRI_BY_ER = [
    clause("procedure", "phrase", "MRI"),
    clause("department", "eq", "emergency"),
    clause("year", "range", "2025~2025"),
    clause("patient_id", "count", "100001~139385"),
]


class TestRendering:
    def test_wire_shape_matches_proxy_search_api(self):
        assert filters_to_wire(parse_filters(MRI_BY_ER)) == [
            {"label": "procedure", "value": "MRI"},
            {"label": "department", "value": "emergency"},
            {"label": "year", "value": "2025~2025"},
            {"label": "patient_id", "value": "count(100001~139385)"},
        ]

    def test_legacy_string(self):
        assert filters_to_legacy_string(parse_filters(MRI_BY_ER)) == (
            "procedure:MRI,department:emergency,year:2025~2025,"
            "patient_id:count(100001~139385)"
        )

    def test_legacy_string_refuses_commas(self):
        with pytest.raises(FilterValidationError, match="contains a comma"):
            filters_to_legacy_string(
                parse_filters([clause("note", "phrase", "chest pain, radiating")])
            )

    def test_explain_reads_as_plain_language(self):
        assert explain_filters(parse_filters(MRI_BY_ER)) == (
            "count of patient_id over 100001~139385 where procedure contains "
            '"MRI" AND department = emergency AND year = 2025'
        )
