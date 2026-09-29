"""blindllm: open-source model instructions for querying encrypted data with an LLM.

The system prompt, the structured-output contract, validation, and adapters for
OpenAI, Anthropic and Gemini. The model only ever sees schemas and aggregate
numbers; records stay encrypted behind the Blind Insight proxy.
"""

from blindllm.contracts import ALLOWED_TOOL_NAMES, OPENAI_STRUCTURED_OUTPUT_SCHEMA
from blindllm.errors import ProviderCallError, classify_provider_exception
from blindllm.prompts import SYSTEM_PROMPT, append_tool_result, build_initial_messages
from blindllm.protocol import ChatClientProtocol, FakeClient
from blindllm.providers import SUPPORTED_PROVIDERS, build_provider, normalize_provider
from blindllm.validation import validate_model_output

__all__ = [
    "ALLOWED_TOOL_NAMES",
    "OPENAI_STRUCTURED_OUTPUT_SCHEMA",
    "SUPPORTED_PROVIDERS",
    "SYSTEM_PROMPT",
    "ChatClientProtocol",
    "FakeClient",
    "ProviderCallError",
    "append_tool_result",
    "build_initial_messages",
    "build_provider",
    "classify_provider_exception",
    "normalize_provider",
    "validate_model_output",
]
__version__ = "0.1.0"
