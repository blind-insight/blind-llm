#!/usr/bin/env python3
"""Has the Blind Insight server's BlindLLM code changed since this package was copied?

Until the server imports this package (instead of keeping its own copies), the
two can drift. This compares the server's files at the extraction commit with
its current ``origin/main`` and lists anything that needs porting here.

    python scripts/check_drift.py --server ../server
"""

from __future__ import annotations

import argparse
import subprocess
import sys

EXTRACTED_FROM = "55fbbe5"  # keep in sync with PROVENANCE.md

FILES = {
    "blindinsight/blindllm/prompts.py": "src/blindllm/prompts.py",
    "blindinsight/blindllm/contracts.py": "src/blindllm/contracts.py",
    "blindinsight/blindllm/validation.py": "src/blindllm/validation.py",
    "blindinsight/blindllm/openai_adapter.py": "src/blindllm/protocol.py",
    "blindinsight/blindllm/provider_errors.py": "src/blindllm/errors.py (lines 1-96)",
    "blindinsight/blindllm/provider_runtime.py": "src/blindllm/runtime.py",
    "blindinsight/blindllm/openai_client.py": "src/blindllm/providers/openai.py",
    "blindinsight/blindllm/anthropic_client.py": "src/blindllm/providers/anthropic.py",
    "blindinsight/blindllm/gemini_client.py": "src/blindllm/providers/gemini.py",
    "blindinsight/blindllm/providers.py": "src/blindllm/providers/__init__.py",
    "blindinsight/blindllm/policy.py": "src/blindllm/guardrails/pii.py",
    "blindinsight/blindllm/result_postprocessing.py": "src/blindllm/guardrails/corrections.py",
    "blindinsight/blindllm/tools.py": "src/blindllm/catalog.py (MOCK_CATALOG, ML models only)",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--server", default="../server", help="path to a server checkout"
    )
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    if not args.no_fetch:
        subprocess.run(["git", "-C", args.server, "fetch", "-q", "origin"], check=False)
    changed = subprocess.run(
        [
            "git",
            "-C",
            args.server,
            "diff",
            "--name-only",
            EXTRACTED_FROM,
            args.ref,
            "--",
            *FILES,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not changed:
        print(
            f"No drift: server {args.ref} matches extraction commit {EXTRACTED_FROM}."
        )
        return 0
    print(f"Server files changed since {EXTRACTED_FROM} — port these:")
    for path in changed:
        print(f"  {path}  →  {FILES[path]}")
    print(f"\nSee: git -C {args.server} diff {EXTRACTED_FROM} {args.ref} -- <file>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
