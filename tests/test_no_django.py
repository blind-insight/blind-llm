"""The package must import without Django (or any web framework) installed."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_pull_in_django():
    code = (
        "import sys, blindllm, blindllm.providers.openai, blindllm.providers.anthropic, "
        "blindllm.providers.gemini, blindllm.guardrails.pii, blindllm.guardrails.corrections, "
        "blindllm.catalog; "
        "assert 'django' not in sys.modules, 'django was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
