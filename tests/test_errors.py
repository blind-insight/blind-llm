"""Tests for normalized BlindLLM provider error handling."""

from __future__ import annotations

from blindllm.errors import (
    ProviderCallError,
    ProviderErrorInfo,
    classify_provider_exception,
)


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "provider failed"):
        super().__init__(message)
        self.status_code = status_code


def test_classify_auth_error():
    err = classify_provider_exception("anthropic", _StatusError(401))
    assert err.info.type == "provider_auth"
    assert "rejected" in err.info.message.lower()


def test_classify_rate_limit_error():
    err = classify_provider_exception("gemini", _StatusError(429))
    assert err.info.type == "provider_rate_limit"


def test_provider_call_error_payload():
    err = ProviderCallError(
        "anthropic",
        ProviderErrorInfo(type="provider_auth", message="Anthropic rejected."),
    )
    assert err.as_payload() == {
        "provider": "anthropic",
        "type": "provider_auth",
        "message": "Anthropic rejected.",
    }


def test_timeout_factory():
    err = ProviderCallError.from_timeout("gemini")
    assert err.info.type == "provider_timeout"
    assert err.provider == "gemini"
