from __future__ import annotations

from types import SimpleNamespace

import pytest

from blindllm.prompts import JSON_ONLY_REMINDER
from blindllm.providers.gemini import GeminiProvider


class _Models:
    def __init__(self, text):
        self.text = text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def _provider(text):
    models = _Models(text)
    return GeminiProvider("g-test", client=SimpleNamespace(models=models)), models


def test_chat_turn_parses_json():
    p, _ = _provider('{"response_type": "final_answer", "narrative": "ok"}')
    assert p.chat_turn([{"role": "user", "content": "q"}])["narrative"] == "ok"


def test_chat_turn_requests_json_mime():
    p, m = _provider("{}")
    p.chat_turn([{"role": "user", "content": "q"}])
    assert m.calls[0]["config"]["response_mime_type"] == "application/json"


def test_chat_turn_does_not_send_unsupported_labels():
    p, m = _provider("{}")
    p.chat_turn([{"role": "user", "content": "q"}], provider_user_id="u-1")
    assert "labels" not in m.calls[0]["config"]


def test_chat_turn_sends_system_instruction_separately():
    p, m = _provider("{}")
    p.chat_turn(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
    )
    call = m.calls[0]
    assert call["config"]["system_instruction"] == "SYS" + JSON_ONLY_REMINDER
    assert call["contents"] == "[USER]\nq"


def test_empty_text_raises():
    p, _ = _provider("")
    with pytest.raises(ValueError):
        p.chat_turn([{"role": "user", "content": "q"}])
