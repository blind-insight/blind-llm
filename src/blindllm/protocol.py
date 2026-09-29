"""The provider-neutral chat client protocol, plus a deterministic fake for tests.

Every provider adapter in ``blindllm.providers`` implements ``ChatClientProtocol``.
(The name ``OpenAIClientProtocol`` is kept as an alias: the protocol started life
as the OpenAI client's shape and the Blind Insight server still imports it.)
"""

from __future__ import annotations

import json
from typing import Any, Protocol


class ChatClientProtocol(Protocol):
    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]: ...

    def build_initial_messages(
        self, prompt: str, context: dict[str, Any]
    ) -> list[dict[str, str]]: ...

    @staticmethod
    def append_tool_result(
        messages: list[dict[str, str]],
        assistant_output: dict[str, Any],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> list[dict[str, str]]: ...


OpenAIClientProtocol = ChatClientProtocol


class OpenAIAdapter:
    """Thin wrapper around any ChatClientProtocol implementation."""

    def __init__(self, client: ChatClientProtocol) -> None:
        self.client = client

    def build_messages(
        self, prompt: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        return self.client.build_initial_messages(prompt=prompt, context=context)

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.client.chat_turn(messages, provider_user_id=provider_user_id)
        if not isinstance(payload, dict):
            raise ValueError("chat client adapter expected a dict response payload")
        return payload

    def append_tool_result(
        self,
        messages: list[dict[str, str]],
        assistant_output: dict[str, Any],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> list[dict[str, str]]:
        return self.client.append_tool_result(
            messages,
            assistant_output,
            tool_name,
            tool_args,
            tool_result,
        )


class FakeClient:
    """Deterministic fake used for tests and local dry-runs: replays scripted responses."""

    def __init__(
        self, scripted_responses: list[dict[str, Any]] | dict[str, Any]
    ) -> None:
        if isinstance(scripted_responses, dict):
            scripted_responses = [scripted_responses]
        self._responses = list(scripted_responses)
        self._call_index = 0
        self.last_messages: list[dict[str, str]] | None = None
        self.last_provider_user_id: str | None = None

    def chat_turn(
        self,
        messages: list[dict[str, str]],
        *,
        provider_user_id: str | None = None,
    ) -> dict[str, Any]:
        self.last_messages = messages
        self.last_provider_user_id = provider_user_id
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return json.loads(json.dumps(resp))

    def build_initial_messages(
        self, prompt: str, context: dict[str, Any]
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": "fake"}]
        if context:
            msgs.append({"role": "system", "content": json.dumps(context)})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def append_tool_result(
        messages: list[dict[str, str]],
        assistant_output: dict[str, Any],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages.append({"role": "assistant", "content": json.dumps(assistant_output)})
        messages.append(
            {
                "role": "user",
                "content": f"Tool {tool_name} result: {json.dumps(tool_result)}",
            }
        )
        return messages


FakeOpenAIClient = FakeClient
