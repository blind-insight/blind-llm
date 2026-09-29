# Provenance

Extracted on 2026-09-28 from the Blind Insight server repository, `blindinsight/blindllm/`, at commit **`55fbbe5`**.

| Server file                                                   | Here                                     | Changes                                                                                                                               |
| ------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `prompts.py`                                                  | `prompts.py`                             | Removed an internal repository path and product-demo wording; dropped an internal keyword; removed a duplicated example with a typo.  |
| `contracts.py`                                                | `contracts.py`                           | None.                                                                                                                                 |
| `validation.py`                                               | `validation.py`                          | Import path.                                                                                                                          |
| `openai_adapter.py`                                           | `protocol.py`                            | `OpenAIClientProtocol` → `ChatClientProtocol` and `FakeOpenAIClient` → `FakeClient`, old names kept as aliases.                       |
| `provider_errors.py` (lines 1-96)                             | `errors.py`                              | Proxy/tool error classification left in the server; quota message made generic.                                                       |
| `provider_runtime.py`                                         | `runtime.py`                             | Timeout is a parameter (`timeout_ms`) instead of a Django setting.                                                                    |
| `openai_client.py`, `anthropic_client.py`, `gemini_client.py` | `providers/{openai,anthropic,gemini}.py` | No Django: `api_key`, `model`, `timeout_ms`, `max_tokens` (Anthropic) are constructor parameters; injectable `client=` and `invoke=`. |
| `providers.py`                                                | `providers/__init__.py`                  | `build_provider(name, api_key, ...)`; front-end-specific aliases and model list removed; `claude` alias added.                        |
| `policy.py`                                                   | `guardrails/pii.py`                      | None.                                                                                                                                 |
| `result_postprocessing.py`                                    | `guardrails/corrections.py`              | None.                                                                                                                                 |
| `tools.py` (`MOCK_CATALOG`, ML approaches)                    | `catalog.py`                             | Copied as sample data; `_DEFAULT_ML_MODELS` → `DEFAULT_ML_MODELS`; one internal keyword removed.                                      |

Tests were ported from `tests/blindllm/` (`test_prompts`, `test_validation`, `test_policy`, `test_result_postprocessing`, provider half of `test_provider_errors`); the provider and factory tests were rewritten to use constructor parameters and injected fake SDK clients.

Until the server imports this package, the two copies can drift. `python scripts/check_drift.py --server ../server` lists server files that changed since `55fbbe5`.
