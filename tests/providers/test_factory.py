from __future__ import annotations

from types import SimpleNamespace

import pytest

from blindllm.providers import (
    ProviderKeyMissing,
    ProviderNotSupported,
    build_provider,
    normalize_provider,
)
from blindllm.providers.anthropic import AnthropicProvider
from blindllm.providers.gemini import GeminiProvider
from blindllm.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "openai"),
        ("", "openai"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("Claude", "anthropic"),
        ("gemini", "gemini"),
        ("google", "gemini"),
        ("  GEMINI ", "gemini"),
    ],
)
def test_normalize(raw, expected):
    assert normalize_provider(raw) == expected


def test_unknown_raises():
    with pytest.raises(ProviderNotSupported):
        normalize_provider("llama")


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("gemini", GeminiProvider),
    ],
)
def test_build_returns_adapter(name, cls):
    p = build_provider(
        name, "key", client=SimpleNamespace(), catalog=[{"dataset": "d", "schema": "s"}]
    )
    assert isinstance(p, cls)
    msgs = p.build_initial_messages("hello", {})
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert "d/s" in msgs[2]["content"]


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
def test_missing_key_raises(name):
    with pytest.raises(ProviderKeyMissing):
        build_provider(name, "")
