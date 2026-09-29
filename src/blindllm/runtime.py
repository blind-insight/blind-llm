"""Run a provider SDK call with a timeout and normalized errors."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import TypeVar

from blindllm.errors import ProviderCallError, classify_provider_exception

T = TypeVar("T")

DEFAULT_TIMEOUT_MS = 60_000

# The signature every provider adapter uses to wrap its SDK call. Pass your own
# (``invoke=...``) to add retries, tracing, or a server's own error handling.
Invoker = Callable[[str, Callable[[], T]], T]


def invoke_provider(
    provider: str,
    fn: Callable[[], T],
    *,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> T:
    """Call ``fn`` with a timeout; map SDK exceptions to ``ProviderCallError``."""
    timeout_s = max(timeout_ms / 1000.0, 1.0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        raise ProviderCallError.from_timeout(provider) from exc
    except ProviderCallError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise classify_provider_exception(provider, exc) from exc
