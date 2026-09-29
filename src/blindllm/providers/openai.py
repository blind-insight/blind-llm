"""OpenAI adapter: ``chat.completions`` with a strict JSON-schema response format."""

from __future__ import annotations

import json
from typing import Any

from blindllm.contracts import OPENAI_STRUCTURED_OUTPUT_SCHEMA
from blindllm.providers._base import BaseProvider, require_key

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIProvider(BaseProvider):
    name = "openai"

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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import-time only
                raise RuntimeError("pip install 'blindllm[openai]'") from exc
            client = OpenAI(api_key=api_key)
        self._client = client

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        _ = provider_user_id

        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bi_response",
                        "strict": True,
                        "schema": OPENAI_STRUCTURED_OUTPUT_SCHEMA,
                    },
                },
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI returned empty content")
            return content

        return json.loads(self._invoke(self.name, _call))
