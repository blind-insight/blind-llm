from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blindllm.providers import ProviderKeyMissing
from blindllm.providers.openai import OpenAIProvider


class _Completions:
    def __init__(self, content):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(content):
    completions = _Completions(content)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_missing_api_key_raises():
    with pytest.raises(ProviderKeyMissing):
        OpenAIProvider(api_key="")


def test_chat_turn_parses_json():
    client, _ = _client(
        json.dumps({"response_type": "final_answer", "narrative": "hi"})
    )
    out = OpenAIProvider("sk-test", client=client).chat_turn(
        [{"role": "user", "content": "x"}]
    )
    assert out["narrative"] == "hi"


def test_chat_turn_passes_strict_json_schema():
    client, completions = _client("{}")
    OpenAIProvider("sk-test", client=client).chat_turn(
        [{"role": "user", "content": "x"}]
    )
    fmt = completions.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "bi_response"


def test_chat_turn_uses_configured_model():
    client, completions = _client("{}")
    OpenAIProvider("sk-test", model="gpt-4o-mini", client=client).chat_turn([])
    assert completions.calls[0]["model"] == "gpt-4o-mini"


def test_empty_content_raises():
    client, _ = _client(None)
    with pytest.raises(ValueError):
        OpenAIProvider("sk-test", client=client).chat_turn([])


def test_custom_invoker_wraps_the_call():
    seen = []

    def invoke(provider, fn):
        seen.append(provider)
        return fn()

    client, _ = _client("{}")
    OpenAIProvider("sk-test", client=client, invoke=invoke).chat_turn([])
    assert seen == ["openai"]
