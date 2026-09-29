"""Normalized provider failure types: map any SDK exception to a stable error shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderErrorInfo:
    type: str
    message: str


class ProviderCallError(RuntimeError):
    """Raised when an upstream LLM provider rejects or fails a call."""

    def __init__(self, provider: str, info: ProviderErrorInfo) -> None:
        self.provider = provider
        self.info = info
        super().__init__(info.message)

    def to_dict(self) -> dict[str, str]:
        return {"type": self.info.type, "message": self.info.message}

    def as_payload(self) -> dict[str, Any]:
        return {"provider": self.provider, **self.to_dict()}

    @classmethod
    def from_timeout(cls, provider: str) -> ProviderCallError:
        label = provider.title()
        return cls(
            provider,
            ProviderErrorInfo(
                type="provider_timeout",
                message=(
                    f"{label} took too long to respond. Please try again in a moment."
                ),
            ),
        )


def _user_message(provider: str, err_type: str) -> str:
    label = provider.title()
    messages = {
        "provider_auth": (
            f"{label} rejected the request. Server logs contain the provider response."
        ),
        "provider_rate_limit": (
            f"{label} rate limit reached. Please wait and try again."
        ),
        "provider_quota": (f"{label} quota exceeded. Please try again later."),
        "provider_payload": (f"{label} could not process the request payload."),
        "provider_model": (f"{label} does not recognize the configured model."),
        "provider_safety": (f"{label} blocked the request for safety reasons."),
        "unknown_provider_error": (
            f"{label} returned an unexpected error. Please try again."
        ),
    }
    return messages.get(err_type, messages["unknown_provider_error"])


def classify_provider_exception(provider: str, exc: Exception) -> ProviderCallError:
    """Map SDK-specific exceptions to a stable public error shape."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    err_type = "unknown_provider_error"
    if status in {401, 403} or "auth" in name or "permission" in name:
        err_type = "provider_auth"
    elif status == 429 or "ratelimit" in name or "rate limit" in text:
        err_type = "provider_rate_limit"
    elif status == 402 or "quota" in text or "billing" in text:
        err_type = "provider_quota"
    elif status == 404 or ("not found" in text and "model" in text):
        err_type = "provider_model"
    elif "safety" in name or "blocked" in text or "content policy" in text:
        err_type = "provider_safety"
    elif status in {400, 422} or "invalid" in name or "badrequest" in name:
        err_type = "provider_payload"

    return ProviderCallError(
        provider,
        ProviderErrorInfo(
            type=err_type,
            message=_user_message(provider, err_type),
        ),
    )
