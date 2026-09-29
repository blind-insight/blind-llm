"""Anthropic adapter: Messages API with system prompts split out, JSON by instruction.

Anthropic's Messages API takes ``system`` separately from ``messages``, so the
leading system turns of the canonical message list are joined into it.
"""

from __future__ import annotations

from typing import Any

from blindllm.prompts import JSON_ONLY_REMINDER
from blindllm.providers._base import BaseProvider, extract_json, require_key

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        if client is None:
            api_key = require_key(self.name, api_key)
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - import-time only
                raise RuntimeError("pip install 'blindllm[anthropic]'") from exc
            client = Anthropic(api_key=api_key)
        self._client = client

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                chat.append(m)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": "\n\n".join(system_parts) + JSON_ONLY_REMINDER,
            "messages": chat,
            "temperature": 0.2,
        }
        if provider_user_id:
            create_kwargs["metadata"] = {"user_id": provider_user_id}

        def _call() -> str:
            response = self._client.messages.create(**create_kwargs)
            text = "".join(
                part.text
                for part in getattr(response, "content", None) or []
                if isinstance(getattr(part, "text", None), str)
            )
            if not text:
                raise ValueError("Anthropic returned empty content")
            return text

        return extract_json(self._invoke(self.name, _call))
