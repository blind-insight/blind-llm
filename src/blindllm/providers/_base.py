"""Shared plumbing for the provider adapters."""

from __future__ import annotations

import functools
import json
import re
from typing import Any

from blindllm.prompts import append_tool_result as _append_tool_result
from blindllm.prompts import build_initial_messages as _build_initial_messages
from blindllm.runtime import DEFAULT_TIMEOUT_MS, Invoker, invoke_provider


class ProviderKeyMissing(RuntimeError):
    """Raised when a provider adapter is constructed without an API key."""


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a completion, tolerating prose or ``` fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


class BaseProvider:
    """Holds the catalog/hints and the invoker; subclasses implement ``chat_turn``."""

    name = "provider"

    def __init__(
        self,
        *,
        catalog: list[dict[str, Any]] | None = None,
        hints: list[str] | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        invoke: Invoker | None = None,
    ) -> None:
        self._catalog = catalog or []
        self._hints = hints or []
        self._invoke: Invoker = invoke or functools.partial(
            invoke_provider, timeout_ms=timeout_ms
        )

    def build_initial_messages(
        self, prompt: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        return _build_initial_messages(prompt, context, self._catalog, self._hints)

    @staticmethod
    def append_tool_result(
        messages: list[dict[str, str]],
        assistant_output: dict[str, Any],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> list[dict[str, str]]:
        return _append_tool_result(
            messages, assistant_output, tool_name, tool_args, tool_result
        )


def require_key(provider: str, api_key: str | None) -> str:
    if not api_key:
        raise ProviderKeyMissing(f"{provider}: api_key is required")
    return api_key
