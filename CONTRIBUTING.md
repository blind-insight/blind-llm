# Contributing

Thanks for helping. A few conventions:

- **Branches and commits.** Work on a branch; use conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
- **Lint and test before you push.** `uv run pre-commit run -a` and `uv run pytest`.
- **No framework dependencies.** The package must import with nothing but the standard library; provider SDKs are optional extras. `tests/test_no_django.py` guards this.

## Adding a provider

1. Create `src/blindllm/providers/<name>.py` with a class that subclasses `BaseProvider` (from `providers/_base.py`) and implements `chat_turn(messages, *, provider_user_id=None) -> dict`.
2. Take `api_key` and `model` in the constructor. Accept `client=` so tests can inject a fake SDK client, and pass `**kwargs` to `BaseProvider` (catalog, hints, timeout_ms, invoke).
3. Wrap the SDK call in `self._invoke(self.name, _call)` so timeouts and error classification are consistent.
4. If the provider has no strict JSON mode, append `JSON_ONLY_REMINDER` to the system prompt and parse with `extract_json`.
5. Register it in `providers/__init__.py` (`SUPPORTED_PROVIDERS`, `PROVIDER_ALIASES`, `build_provider`) and add an optional extra in `pyproject.toml`.
6. Add `tests/providers/test_<name>.py` using an injected fake client (see the existing tests).

## Changing the prompt

The system prompt is shared by every provider. Changes affect all of them, so include a before/after example of the model's tool call in your pull request.
