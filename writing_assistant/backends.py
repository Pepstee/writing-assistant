from __future__ import annotations

import subprocess
from collections import deque


class MockBackend:
    """Fully deterministic backend for testing; cycles through a fixed response list."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: deque[str] = deque(responses or ["mock response"])

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        response = self._responses[0]
        self._responses.rotate(-1)
        return response


class ClaudeCLIBackend:
    """LLM backend that calls the `claude` CLI via subprocess and returns its stdout."""

    def __init__(self, model: str = "claude-sonnet-4-6", extra_args: list[str] | None = None) -> None:
        self.model = model
        self.extra_args = extra_args or []

    def complete(self, prompt: str) -> str:
        cmd = ["claude", "--model", self.model, *self.extra_args, "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
