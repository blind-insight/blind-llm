"""Azure OpenAI adapter: Chat Completions against a customer's Azure deployment.

Azure OpenAI (and Azure AI Foundry's OpenAI endpoint) serve the same Chat
Completions API as OpenAI under the customer's own resource. Two things
differ:

- ``model`` is the customer's **deployment name**, not a model id.
- The endpoint is the customer's resource URL.

By default this uses Azure's **v1 API**, which Microsoft recommends since
August 2025: the plain ``OpenAI`` client with ``base_url`` set to
``<endpoint>/openai/v1/`` and no dated ``api-version``. Pass ``api_version=``
(for example ``"2024-10-21"``) only for a resource or gateway that still needs
the older dated API; that uses the ``AzureOpenAI`` client instead.

Authentication is by API key only, always passed to the SDK explicitly, so
the SDK never falls back to an Entra ID token or another key found in the
environment.

Each setting falls back to an environment variable: ``AZURE_OPENAI_API_KEY``,
``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_DEPLOYMENT``.

Not every deployment or API version enforces a JSON schema, so the output mode
is configurable:

- ``json_schema`` (default): strict ``json_schema`` response format.
- ``json_object``: JSON mode; valid JSON is guaranteed, the shape is not.
- ``prompt``: no response format; JSON by instruction, as for Anthropic.

The two weaker modes add the JSON-only reminder and parse leniently; callers
validate the result with ``validate_model_output`` either way.
"""

from __future__ import annotations

import json
import os
from typing import Any

from blindllm.contracts import OPENAI_STRUCTURED_OUTPUT_SCHEMA
from blindllm.errors import ProviderCallError
from blindllm.prompts import JSON_ONLY_REMINDER
from blindllm.providers._base import (
    BaseProvider,
    ProviderKeyMissing,
    extract_json,
    require_key,
)

STRUCTURED_OUTPUT_MODES: tuple[str, ...] = ("json_schema", "json_object", "prompt")


class AzureOpenAIProvider(BaseProvider):
    name = "azure"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        endpoint: str | None = None,
        api_version: str | None = None,
        structured_output: str = "json_schema",
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"structured_output must be one of {STRUCTURED_OUTPUT_MODES}, "
                f"got {structured_output!r}"
            )
        deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ValueError(
                "azure: a deployment name is required "
                "(model= or AZURE_OPENAI_DEPLOYMENT)"
            )
        self._model = deployment
        self._structured_output = structured_output
        if client is None:
            api_key = require_key(
                self.name, api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            )
            endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ProviderKeyMissing(
                    "azure: an endpoint is required (endpoint= or AZURE_OPENAI_ENDPOINT)"
                )
            client = _make_client(api_key, endpoint, api_version)
        self._client = client

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        _ = provider_user_id
        request: dict[str, Any] = {"model": self._model, "temperature": 0.2}
        if self._structured_output == "json_schema":
            request["messages"] = messages
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "bi_response",
                    "strict": True,
                    "schema": OPENAI_STRUCTURED_OUTPUT_SCHEMA,
                },
            }
        else:
            request["messages"] = _with_json_reminder(messages)
            if self._structured_output == "json_object":
                request["response_format"] = {"type": "json_object"}

        def _call() -> str:
            response = self._client.chat.completions.create(**request)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "content_filter":
                # Azure's content filter withheld the reply.
                raise ProviderCallError.of_type(self.name, "provider_safety")
            content = choice.message.content
            if content is None:
                raise ValueError("Azure OpenAI returned empty content")
            return content

        text = self._invoke(self.name, _call)
        if self._structured_output == "json_schema":
            return json.loads(text)
        return extract_json(text)


def v1_base_url(endpoint: str) -> str:
    """``https://<resource>.openai.azure.com`` → ``.../openai/v1/`` (idempotent)."""
    url = endpoint.rstrip("/")
    if not url.endswith("/openai/v1"):
        url += "/openai/v1"
    return url + "/"


def _make_client(api_key: str, endpoint: str, api_version: str | None) -> Any:
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:  # pragma: no cover - import-time only
        raise RuntimeError("pip install 'blindllm[openai]'") from exc
    if api_version:
        # The older dated API: /openai/deployments/<deployment>?api-version=...
        return AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )
    return OpenAI(api_key=api_key, base_url=v1_base_url(endpoint))


def _with_json_reminder(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append the JSON-only reminder to the first system message (copying it)."""
    out = [dict(m) for m in messages]
    for m in out:
        if m["role"] == "system":
            m["content"] += JSON_ONLY_REMINDER
            return out
    return [{"role": "system", "content": JSON_ONLY_REMINDER.strip()}, *out]
