"""What the model sees, and what it sends back.

By default this prints the instruction messages for a question and replays a
scripted model reply through validation. With ``--live`` (and the provider's API
key in the environment) it asks a real model which tool it would call first. No
data is touched either way:

    python examples/quickstart.py "average risk for German accounts in 2024?"
    ANTHROPIC_API_KEY=... python examples/quickstart.py --live --provider anthropic "..."
    GOOGLE_API_KEY= python examples/quickstart.py --live --provider gemini
"""

from __future__ import annotations

import argparse
import json
import os

from blindllm import (
    FakeClient,
    ProviderCallError,
    build_provider,
    explain_filters,
    parse_filters,
    validate_model_output,
)
from blindllm.catalog import MOCK_CATALOG
from blindllm.prompts import build_initial_messages

KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "question",
        nargs="?",
        default="How many MRI scans did the ER department order in March 2025?",
    )
    p.add_argument("--provider", default="openai", choices=sorted(KEY_ENV))
    p.add_argument("--live", action="store_true", help="call the real provider")
    args = p.parse_args()

    catalog = [
        {
            k: v
            for k, v in e.items()
            if k in ("dataset", "schema", "label", "description", "fields")
        }
        for e in MOCK_CATALOG
    ]
    key = os.environ.get(KEY_ENV[args.provider], "") if args.live else ""
    if args.live and not key:
        raise SystemExit(f"--live needs {KEY_ENV[args.provider]} in the environment")

    if key:
        client = build_provider(args.provider, key, catalog=catalog)
        messages = client.build_initial_messages(args.question, {})
    else:
        messages = build_initial_messages(args.question, {}, catalog, [])
        # What a model should send for the default question: a phrase clause on
        # the free-text procedure field, scoped by equality, counted on the
        # schema's count vehicle.
        client = FakeClient(
            {
                "response_type": "tool_call",
                "tool_name": "query_aggregate",
                "tool_args": {
                    "dataset": "population-health",
                    "schema": "ehr-data-v3",
                    "filters": [
                        {"field": "procedure", "op": "phrase", "value": "MRI"},
                        {"field": "department", "op": "eq", "value": "emergency"},
                        {"field": "year", "op": "eq", "value": "2025"},
                        {"field": "month", "op": "eq", "value": "3"},
                        {
                            "field": "patient_id",
                            "op": "count",
                            "value": "100001~139385",
                        },
                    ],
                },
                "narrative": None,
                "citations": None,
            }
        )
        print("(scripted reply; pass --live to ask a real model)\n")

    print("=== messages sent to the model ===")
    for m in messages:
        body = m["content"] if len(m["content"]) < 400 else m["content"][:400] + " …"
        print(f"[{m['role']}] {body}\n")

    try:
        reply = client.chat_turn(messages)
    except ProviderCallError as exc:
        raise SystemExit(f"{exc.provider}: {exc.info.type}: {exc}") from exc
    print("=== model reply ===")
    print(json.dumps(reply, indent=2))
    # Raises ValueError if the reply breaks the contract or names a field/op the
    # catalog's schema does not support.
    validate_model_output(reply, catalog=catalog)
    print("\n=== validation: reply matches the output contract ===")
    if reply.get("tool_name") == "query_aggregate":
        clauses = parse_filters(reply["tool_args"]["filters"])
        print(f"\n=== what will be asked ===\n{explain_filters(clauses)}")


if __name__ == "__main__":
    main()
