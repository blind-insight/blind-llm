from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from blindllm.errors import ProviderCallError
from blindllm.providers import ProviderKeyMissing, build_provider, normalize_provider
from blindllm.providers.azure import AzureOpenAIProvider, v1_base_url

AZURE_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_AD_TOKEN",
    "AZURE_OPENAI_DEPLOYMENT",
    "OPENAI_API_VERSION",
)
ENDPOINT = "https://contoso.openai.azure.com"
V1_URL = "https://contoso.openai.azure.com/openai/v1/"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in AZURE_ENV:
        monkeypatch.delenv(name, raising=False)


class _Completions:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


def _client(content, finish_reason="stop"):
    completions = _Completions(content, finish_reason)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.fixture
def fake_sdk(monkeypatch):
    """A stand-in ``openai`` module recording which client was built, and how."""
    created: list[tuple[str, dict]] = []

    class OpenAI:
        def __init__(self, **kwargs):
            created.append(("OpenAI", kwargs))

    class AzureOpenAI:
        def __init__(self, **kwargs):
            created.append(("AzureOpenAI", kwargs))

    module = ModuleType("openai")
    module.OpenAI = OpenAI
    module.AzureOpenAI = AzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return created


# -- configuration ----------------------------------------------------------


def test_deployment_is_required():
    with pytest.raises(ValueError, match="deployment name is required"):
        AzureOpenAIProvider("key", client=SimpleNamespace())


def test_endpoint_is_required_without_a_client(fake_sdk):
    with pytest.raises(ProviderKeyMissing, match="endpoint is required"):
        AzureOpenAIProvider("key", "my-deployment")


def test_defaults_to_the_v1_api_with_no_api_version(fake_sdk):
    AzureOpenAIProvider("key", "my-deployment", endpoint=ENDPOINT)
    assert fake_sdk == [("OpenAI", {"api_key": "key", "base_url": V1_URL})]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://contoso.openai.azure.com",
        "https://contoso.openai.azure.com/",
        "https://contoso.openai.azure.com/openai/v1",
        "https://contoso.openai.azure.com/openai/v1/",
    ],
)
def test_v1_base_url_accepts_the_resource_or_the_v1_url(endpoint):
    assert v1_base_url(endpoint) == V1_URL


def test_an_explicit_api_version_uses_the_dated_api(fake_sdk):
    AzureOpenAIProvider("key", "dep", endpoint=ENDPOINT, api_version="2024-10-21")
    assert fake_sdk == [
        (
            "AzureOpenAI",
            {"api_key": "key", "azure_endpoint": ENDPOINT, "api_version": "2024-10-21"},
        )
    ]


def test_settings_fall_back_to_the_environment(fake_sdk, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    # A leftover OPENAI_API_VERSION does not silently switch to the dated API.
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-01-01")
    provider = AzureOpenAIProvider()
    # The key is read here and always passed explicitly, never left to the SDK.
    assert fake_sdk == [("OpenAI", {"api_key": "env-key", "base_url": V1_URL})]
    assert provider._model == "env-deployment"


def test_missing_api_key_raises(fake_sdk):
    with pytest.raises(ProviderKeyMissing):
        AzureOpenAIProvider("", "dep", endpoint=ENDPOINT)
    assert fake_sdk == []


def test_an_entra_id_token_in_the_environment_is_never_used(fake_sdk, monkeypatch):
    # API keys only: an AD token alone must not authenticate the client.
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "entra-token")
    with pytest.raises(ProviderKeyMissing):
        AzureOpenAIProvider(None, "dep", endpoint=ENDPOINT)


def test_entra_id_options_are_not_accepted():
    with pytest.raises(TypeError):
        AzureOpenAIProvider(
            "key",
            "dep",
            client=SimpleNamespace(),
            azure_ad_token_provider=lambda: "t",
        )


# -- requests ---------------------------------------------------------------


def test_deployment_is_sent_as_the_model_with_a_strict_schema():
    client, completions = _client(json.dumps({"response_type": "final_answer"}))
    out = AzureOpenAIProvider("key", "my-deployment", client=client).chat_turn([])
    call = completions.calls[0]
    assert call["model"] == "my-deployment"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert out == {"response_type": "final_answer"}


def test_json_object_mode_adds_the_reminder_and_parses_leniently():
    client, completions = _client('```json\n{"response_type": "tool_call"}\n```')
    provider = AzureOpenAIProvider(
        "key", "dep", client=client, structured_output="json_object"
    )
    messages = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    out = provider.chat_turn(messages)
    call = completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["content"].startswith("base")
    assert "ONE JSON object only" in call["messages"][0]["content"]
    assert messages[0]["content"] == "base"  # the caller's list is not mutated
    assert out == {"response_type": "tool_call"}


def test_prompt_mode_sends_no_response_format():
    client, completions = _client('{"response_type": "final_answer"}')
    AzureOpenAIProvider(
        "key", "dep", client=client, structured_output="prompt"
    ).chat_turn([{"role": "user", "content": "q"}])
    call = completions.calls[0]
    assert "response_format" not in call
    assert call["messages"][0]["role"] == "system"


def test_unknown_structured_output_mode_raises():
    with pytest.raises(ValueError, match="structured_output"):
        AzureOpenAIProvider(
            "key", "dep", client=SimpleNamespace(), structured_output="xml"
        )


def test_filtered_reply_is_a_safety_error_not_a_parse_error():
    client, _ = _client(None, finish_reason="content_filter")
    with pytest.raises(ProviderCallError) as info:
        AzureOpenAIProvider("key", "dep", client=client).chat_turn([])
    assert info.value.info.type == "provider_safety"
    assert info.value.provider == "azure"


# -- factory ----------------------------------------------------------------


@pytest.mark.parametrize(
    "alias", ["azure", "Azure-OpenAI", "azure_openai", "microsoft"]
)
def test_aliases_normalize_to_azure(alias):
    assert normalize_provider(alias) == "azure"


def test_build_provider_passes_deployment_and_settings(fake_sdk):
    provider = build_provider(
        "azure", "key", model="dep", endpoint=ENDPOINT, api_version="2024-10-21"
    )
    assert isinstance(provider, AzureOpenAIProvider)
    assert provider._model == "dep"
    assert fake_sdk[0] == (
        "AzureOpenAI",
        {"api_key": "key", "azure_endpoint": ENDPOINT, "api_version": "2024-10-21"},
    )
