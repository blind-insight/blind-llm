# blindllm

**Open-source model instructions for asking an LLM questions about encrypted data.**

`blindllm` holds the parts of [Blind Insight](https://blindinsight.com)'s BlindLLM that tell a model _what to do_: the system prompt, the structured-output contract, validation, and adapters for OpenAI, Anthropic and Gemini. The model is taught to answer questions with **encrypted aggregates only**. It sees schemas and numbers, never records.

```
question ──▶ blindllm: instructions + provider adapter ──▶ any model
                                                              │  tool call (JSON)
                                                              ▼
            your orchestrator ──▶ Blind Insight proxy (holds keys) ──▶ encrypted index
                                                              │  one number
                                                              ▼
                                                        final answer
```

What stays out of this package: running the tools, the orchestration loop, credentials, and the proxy. Those belong to the product (or to your own orchestrator). This package is the instruction layer.

## Install

```bash
pip install -e ".[anthropic]"     # or [openai], [gemini], [all]  (PyPI release to follow)
```

## Quickstart

```python
from blindllm import build_provider, validate_model_output

catalog = [
    {"dataset": "fraud-data", "schema": "train", "label": "Fraud account records"}
]
client = build_provider("anthropic", api_key="...", catalog=catalog)

messages = client.build_initial_messages(
    "Average risk for German accounts in 2024?", {}
)
reply = client.chat_turn(
    messages
)  # {'response_type': 'tool_call', 'tool_name': 'describe_schema', ...}
validate_model_output(reply, catalog=catalog)  # raises if the model broke the contract

# run the tool yourself, then feed the result back:
messages = client.append_tool_result(
    messages, reply, reply["tool_name"], reply["tool_args"], tool_result
)
reply = client.chat_turn(messages)  # ... until response_type == "final_answer"
```

`python examples/quickstart.py` prints exactly what the model receives. Add `--live` to ask a real model.

## The contract

The model may call four tools. None of them returns a record.

| Tool                  | Purpose                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| `list_schemas`        | Which datasets and schemas exist                                          |
| `describe_schema`     | Fields, which query operations each supports, suggested ML approaches     |
| `query_aggregate`     | One encrypted aggregate: `count`, `avg`, `sum`, `min`, `max` with filters |
| `suggest_ml_approach` | Pre-vetted models that train from encrypted aggregates                    |

Every reply is one JSON object matching `OPENAI_STRUCTURED_OUTPUT_SCHEMA`:

```json
{
  "response_type": "tool_call",
  "tool_name": "query_aggregate",
  "tool_args": {
    "dataset": "fraud-data",
    "schema": "train",
    "filters": [
      { "field": "year", "op": "eq", "value": "2024" },
      { "field": "account_jurisdiction", "op": "eq", "value": "DE" },
      { "field": "risk_level", "op": "avg", "value": "0~100" }
    ]
  },
  "narrative": null,
  "citations": null
}
```

`filters` is a list of clauses joined by AND. Each clause has an `op`:

| `op`                                  | Meaning                                              | Fields                       |
| ------------------------------------- | ---------------------------------------------------- | ---------------------------- |
| `eq`                                  | Whole-value equality, exact and case-sensitive       | plain strings, integers      |
| `phrase`                              | The text contains these words, consecutively, in order | strings marked `text: true`  |
| `range`                               | Inclusive integer range, `lo~hi`                     | integers                     |
| `count` / `avg` / `sum` / `min` / `max` | The query's one aggregate, over `lo~hi` or `>40`     | integers                     |

The system prompt teaches the model to normalize everyday language ("between 2022 and 2024", "Germany", "the ER", "of those") into these clauses. See `SYSTEM_PROMPT` in [`prompts.py`](src/blindllm/prompts.py).

`validate_model_output(reply, catalog=catalog)` rejects a malformed clause, a second aggregate, and a phrase that normalizes to zero tokens. When the catalog lists the schema's `fields`, it also rejects unknown fields, ops a field doesn't support (`eq` on a `text: true` field, `phrase` on a plain string), and `eq` values outside a field's `values`.

## Phrase search

Fields created with `text: true` (clinical notes, procedure and diagnosis lists) are searched by phrase:

```json
{ "field": "procedure", "op": "phrase", "value": "MRI" }
```

This matches "MRI brain" and "MRI knee". Matching mirrors the proxy's tokenizer (`blindllm.filters.tokenize`): lowercase, split on anything outside `a-z0-9`, consecutive tokens in order, no stemming, no stopword removal, no wildcards. A phrase clause composes with equality, range and one aggregate in a single query, so "How many MRI scans did the ER order in March 2025?" is one encrypted count:

```
count of patient_id over 100001~139385 where procedure contains "MRI"
  AND department = emergency AND year = 2025 AND month = 3
```

`explain_filters` produces that line, and the prompt asks the model to put one per query in `citations`, so the user can check how an answer was computed.

Things to know before relying on it:

- **A phrase with no tokens is a data-exposure bug, not an empty query.** The proxy drops such a filter and returns every record the caller can read (BAC-537). `parse_filters` refuses it before it reaches your executor.
- **Phrase queries are slow:** 8–12 s warm, around 40 s cold, on a 10k-record schema.
- **`text: true` leaks more than a plain string:** the server stores token positions. Enable it per field, deliberately.
- Needs proxy v10.4.0+ and server v10.21.0+. Older versions return empty results with no error.

To run a query, pass `filters_to_wire(parse_filters(...))` to the proxy's `/api/records/search/`. Executors that still take the old comma-joined filter string can use `filters_to_legacy_string`, which refuses values containing a comma.

## Providers

All three adapters implement the same `ChatClientProtocol` and share one canonical message list (system → date → catalog → hints → user). Only the SDK boundary differs:

| Provider  | Structured output                                   | Default model       |
| --------- | --------------------------------------------------- | ------------------- |
| OpenAI    | strict `json_schema` response format                | `gpt-4.1-mini`      |
| Anthropic | system prompt split out, JSON by instruction        | `claude-sonnet-4-6` |
| Gemini    | `system_instruction` + `application/json` MIME type | `gemini-3.1-flash-lite`  |

Defaults match what the Blind Insight service runs today. Pass `model=` to use something newer. Check that the default is still served before relying on it.

Every adapter takes `timeout_ms=` and an optional `invoke=` callable that wraps the SDK call (add retries, tracing, or your own error handling). Provider failures come back as `ProviderCallError` with a stable `type`: `provider_auth`, `provider_rate_limit`, `provider_quota`, `provider_timeout`, and so on.

## Guardrails

- `blindllm.guardrails.pii`: rejects tool payloads and narratives that contain PII patterns (emails, card-like numbers, IBAN-like strings).
- `blindllm.guardrails.corrections`: detects answers the model computed the wrong way (for example, an average built from separate sum and count queries) and builds the corrective message to send back mid-conversation.

## Deciding what the model may see

`blindllm` teaches the model how to ask. To decide _what_ it may ask about (per role, purpose and regulation), and to turn that into encryption keys, pair it with [`blind-policy`](https://github.com/blind-insight). Show the model only the fields in `plan.visible`, and check every `query_aggregate` filter against `plan.queryable`.

## Develop

```bash
uv sync --all-extras
uv run pytest
uv run pre-commit run -a
```

This package was extracted from the Blind Insight server. See [`PROVENANCE.md`](PROVENANCE.md) for the file map, and run `scripts/check_drift.py` to see whether the server has changed since.

## License

MIT.
