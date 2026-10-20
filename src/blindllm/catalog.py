"""Sample schema catalog and ML approach descriptions (what the model sees).

Copied from the Blind Insight server's BlindLLM tools module so the
instructions in this package can be exercised without a live catalog. Pass
your own catalog (from ``list_schemas`` / ``describe_schema``) in production.
"""

from __future__ import annotations

from typing import Any

# Pre-vetted ML approaches reused across catalog entries. Each schema
# can override `ml_models` if it needs a different mix.
DEFAULT_ML_MODELS: list[dict[str, Any]] = [
    {
        "id": "naive-bayes",
        "label": "Categorical Naive Bayes",
        "best_for": (
            "binary or multi-class classification when most features are "
            "categorical (jurisdiction, type, density bucket, etc.)"
        ),
        "query_budget": "~90 aggregate counts (one per feature value per class)",
        "reference": "ml-examples/blind_ml.py (CategoricalNB)",
    },
    {
        "id": "decision-tree",
        "label": "Decision Tree (Gini / CART)",
        "best_for": (
            "rule-based classification with auditable splits; depth ≤3 "
            "stays fast and explainable"
        ),
        "query_budget": "reuses Naive Bayes counts; 0 extra encrypted queries",
        "reference": "ml-examples/blind_ml.py (DecisionTree)",
    },
    {
        "id": "logistic-regression",
        "label": "Logistic Regression (OLS seed + IRLS)",
        "best_for": "calibrated probabilities and a linear decision boundary",
        "query_budget": "reuses Naive Bayes counts; refined locally",
        "reference": "ml-examples/blind_ml.py (LogisticRegression)",
    },
]

MOCK_CATALOG: list[dict[str, Any]] = [
    {
        "dataset": "fraud-data",
        "schema": "train",
        "label": "Fraud account records",
        "keywords": [
            "fraud",
            "risk",
            "money",
            "transaction",
            "transfer",
            "account",
            "mule",
            "card",
            "identity",
            "jurisdiction",
            "bank",
            "aml",
        ],
        "description": (
            "Account-level fraud signals across jurisdictions. Use encrypted "
            "aggregates (count / avg / sum / min / max) to answer rate, "
            "segment, and trend questions; cite an ML approach when the "
            "question is predictive."
        ),
        # Numeric indexed field + safe range to use when you need to count
        # records that match an equality or range filter:
        #   risk_level:count(0~100),name:Bob
        "count_vehicle": {"field": "risk_level", "range": "0~100"},
        "fields": [
            {
                "name": "risk_level",
                "type": "int",
                "indexed": True,
                "range": "0-100",
                "purpose": "fraud risk score; >=50 is high-risk",
            },
            {
                "name": "account_jurisdiction",
                "type": "string",
                "indexed": True,
                "values": [
                    "JP",
                    "AU",
                    "DE",
                    "FR",
                    "US",
                    "GB",
                    "BR",
                    "ES",
                    "HK",
                    "CH",
                    "SG",
                    "CA",
                ],
            },
            {
                "name": "fraud_type",
                "type": "string",
                "indexed": True,
                "values": [
                    "mule_account",
                    "card_fraud",
                    "identity_theft",
                    "account_takeover",
                    "synthetic_identity",
                    "suspicious_transfer",
                    "unusual_activity",
                ],
            },
            {
                "name": "year",
                "type": "int",
                "indexed": True,
                "range": "2021-2025",
                "purpose": (
                    "fraud report year; numeric so range filters like "
                    "`year:2021~2025` scope other aggregates"
                ),
            },
        ],
        "ml_models": DEFAULT_ML_MODELS,
    },
    {
        "dataset": "breast-cancer-research",
        "schema": "screening",
        "label": "Breast cancer screening data",
        "keywords": [
            "health",
            "healthcare",
            "cancer",
            "screening",
            "mammogram",
            "breast",
            "density",
            "risk factor",
            "patient",
            "cohort",
        ],
        "description": (
            "Screening cohort data for prevalence, relative-risk, and "
            "predictive questions. Aggregates run on encrypted records; "
            "ML recipes mirror the healthcare examples in ml-examples/."
        ),
        "count_vehicle": {"field": "cancer_5yr", "range": "0~3"},
        "fields": [
            {
                "name": "cancer_5yr",
                "type": "int",
                "indexed": True,
                "range": "0-3",
                "purpose": "5-year cancer outcome bucket; >=1 indicates a case",
            },
            {
                "name": "age_group",
                "type": "string",
                "indexed": True,
                "values": ["40_49", "50_59", "60_69", "70_74"],
            },
            {
                "name": "race_ethnicity",
                "type": "string",
                "indexed": True,
                "values": ["white", "black", "hispanic", "asian_pi", "other"],
            },
            {
                "name": "breast_density",
                "type": "string",
                "indexed": True,
                "values": ["1", "2", "3", "4"],
                "purpose": "BI-RADS density (1=fatty .. 4=extremely dense)",
            },
            {
                "name": "family_history",
                "type": "string",
                "indexed": True,
                "values": ["yes", "no"],
            },
        ],
        "ml_models": DEFAULT_ML_MODELS,
    },
    {
        # Synthetic encounters with free text: `text: true` fields take phrase
        # search, plain strings take equality. Mirrors the EHR demo's
        # `ehr-data-v3` schema.
        "dataset": "population-health",
        "schema": "ehr-data-v3",
        "label": "EHR encounter records",
        "keywords": [
            "health",
            "hospital",
            "ehr",
            "emr",
            "encounter",
            "visit",
            "admission",
            "department",
            "procedure",
            "scan",
            "imaging",
            "diagnosis",
            "readmission",
            "clinical note",
            "symptom",
        ],
        "description": (
            "Hospital encounters: department, procedures, diagnoses, cost, "
            "length of stay, 30-day readmission, and a free-text clinical "
            "note. Symptoms and imaging phrasing appear only in the note."
        ),
        "count_vehicle": {"field": "patient_id", "range": "100001~139385"},
        "fields": [
            {
                "name": "patient_id",
                "type": "int",
                "indexed": True,
                "range": "100001-139385",
                "purpose": "count vehicle; stable across a patient's visits",
            },
            {"name": "year", "type": "int", "indexed": True, "range": "2022-2026"},
            {"name": "month", "type": "int", "indexed": True, "range": "1-12"},
            {"name": "day", "type": "int", "indexed": True, "range": "1-31"},
            {"name": "age", "type": "int", "indexed": True, "range": "18-94"},
            {
                "name": "sex",
                "type": "string",
                "indexed": True,
                "values": ["female", "male"],
            },
            {
                "name": "department",
                "type": "string",
                "indexed": True,
                "values": [
                    "cardiology",
                    "emergency",
                    "infectious disease",
                    "oncology",
                    "orthopedics",
                    "primary care",
                    "pulmonology",
                    "radiology",
                ],
                "purpose": "ER / ED / emergency room → emergency",
            },
            {
                "name": "procedure",
                "type": "string",
                "text": True,
                "indexed": True,
                "purpose": (
                    "comma-separated procedures ordered, e.g. 'MRI brain', "
                    "'CT chest', 'ECG', 'complete blood count'"
                ),
            },
            {
                "name": "diagnosis",
                "type": "string",
                "text": True,
                "indexed": True,
                "purpose": (
                    "comma-separated diagnoses, e.g. 'urinary tract infection', "
                    "'acute myocardial infarction', 'Lyme disease'"
                ),
            },
            {
                "name": "billed_amount",
                "type": "int",
                "indexed": True,
                "range": "0-39973",
                "purpose": "encounter cost in whole dollars",
            },
            {
                "name": "length_of_stay_days",
                "type": "int",
                "indexed": True,
                "range": "0-8",
            },
            {
                "name": "readmitted_30d",
                "type": "int",
                "indexed": True,
                "range": "0-1",
                "purpose": "1 if readmitted within 30 days",
            },
            {
                "name": "note",
                "type": "string",
                "text": True,
                "indexed": True,
                "purpose": (
                    "free-text clinical note: chief complaint, symptoms, "
                    "findings, disposition"
                ),
            },
        ],
        "ml_models": DEFAULT_ML_MODELS,
    },
]
