from __future__ import annotations

from types import SimpleNamespace

import pytest

from blindllm.prompts import JSON_ONLY_REMINDER
from blindllm.providers.anthropic import AnthropicProvider


class _Messages:
    def __init__(self, text):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text)] if self.text else []
        )


def _provider(text, **kw):
    messages = _Messages(text)
    return AnthropicProvider(
        "ant-test", client=SimpleNamespace(messages=messages), **kw
    ), messages


def test_chat_turn_parses_clean_json():
    p, _ = _provider('{"response_type": "final_answer", "narrative": "ok"}')
    assert p.chat_turn([{"role": "user", "content": "q"}])["narrative"] == "ok"


def test_chat_turn_extracts_json_from_fenced_response():
    p, _ = _provider(
        'Sure:\n```json\n{"response_type": "final_answer", "narrative": "x"}\n```'
    )
    assert p.chat_turn([{"role": "user", "content": "q"}])["narrative"] == "x"


def test_chat_turn_passes_metadata_user_id():
    p, m = _provider("{}")
    p.chat_turn([{"role": "user", "content": "q"}], provider_user_id="u-1")
    assert m.calls[0]["metadata"] == {"user_id": "u-1"}


def test_chat_turn_separates_system_from_messages():
    p, m = _provider("{}")
    p.chat_turn(
        [
            {"role": "system", "content": "SYS-A"},
            {"role": "system", "content": "SYS-B"},
            {"role": "user", "content": "q"},
        ]
    )
    call = m.calls[0]
    assert call["system"] == "SYS-A\n\nSYS-B" + JSON_ONLY_REMINDER
    assert call["messages"] == [{"role": "user", "content": "q"}]


def test_max_tokens_is_a_constructor_param():
    p, m = _provider("{}", max_tokens=512)
    p.chat_turn([{"role": "user", "content": "q"}])
    assert m.calls[0]["max_tokens"] == 512


def test_empty_content_raises():
    p, _ = _provider("")
    with pytest.raises(ValueError):
        p.chat_turn([{"role": "user", "content": "q"}])
