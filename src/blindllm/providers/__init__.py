"""Provider factory: one entry point from a provider id to a ChatClientProtocol client."""

from __future__ import annotations

from typing import Any

from blindllm.protocol import ChatClientProtocol
from blindllm.providers._base import ProviderKeyMissing

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"

SUPPORTED_PROVIDERS: tuple[str, ...] = (
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
)

# Accepted spellings → provider id.
PROVIDER_ALIASES: dict[str, str] = {
    PROVIDER_OPENAI: PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC: PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI: PROVIDER_GEMINI,
    "google": PROVIDER_GEMINI,
    "claude": PROVIDER_ANTHROPIC,
}


class ProviderNotSupported(ValueError):
    """Raised for a provider id this package does not ship."""


def normalize_provider(name: str | None) -> str:
    """Map a provider id or alias to a supported provider. Empty → OpenAI."""
    if not name:
        return PROVIDER_OPENAI
    key = name.strip().lower()
    if key in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[key]
    raise ProviderNotSupported(
        f"provider {name!r} is not supported. Pick one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def build_provider(
    name: str | None,
    api_key: str,
    *,
    model: str | None = None,
    catalog: list[dict[str, Any]] | None = None,
    hints: list[str] | None = None,
    **kwargs: Any,
) -> ChatClientProtocol:
    """Construct a provider adapter. ``kwargs`` pass through (timeout_ms, invoke, max_tokens)."""
    provider = normalize_provider(name)
    common: dict[str, Any] = {"catalog": catalog, "hints": hints, **kwargs}
    if model:
        common["model"] = model
    if provider == PROVIDER_ANTHROPIC:
        from blindllm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key, **common)
    if provider == PROVIDER_GEMINI:
        from blindllm.providers.gemini import GeminiProvider

        return GeminiProvider(api_key, **common)
    from blindllm.providers.openai import OpenAIProvider

    return OpenAIProvider(api_key, **common)


__all__ = [
    "PROVIDER_ALIASES",
    "SUPPORTED_PROVIDERS",
    "ProviderKeyMissing",
    "ProviderNotSupported",
    "build_provider",
    "normalize_provider",
]
