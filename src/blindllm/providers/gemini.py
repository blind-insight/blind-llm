"""Google Gemini adapter: ``generate_content`` with a system instruction and JSON MIME type."""

from __future__ import annotations

from typing import Any

from blindllm.prompts import JSON_ONLY_REMINDER
from blindllm.providers._base import BaseProvider, extract_json, require_key

DEFAULT_MODEL = "gemini-2.0-flash"


def split_system_and_contents(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Render canonical messages across Gemini's system/content boundary.

    Non-system turns stay role-labeled in ``contents`` so tool-call history
    stays legible without asking Gemini to treat system text as user text.
    """
    system_parts: list[str] = []
    content_parts: list[str] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            content_parts.append(f"[{m['role'].upper()}]\n{m['content']}")
    return "\n\n".join(system_parts) + JSON_ONLY_REMINDER, "\n\n".join(content_parts)


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model or DEFAULT_MODEL
        if client is None:
            api_key = require_key(self.name, api_key)
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - import-time only
                raise RuntimeError("pip install 'blindllm[gemini]'") from exc
            client = genai.Client(api_key=api_key)
        self._client = client

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        system_instruction, contents = split_system_and_contents(messages)
        config: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }
        # The Gemini Developer API (API key) does not support request labels.
        _ = provider_user_id

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("Gemini returned empty content")
            return text

        return extract_json(self._invoke(self.name, _call))
