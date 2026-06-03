from __future__ import annotations

import shutil
import subprocess
from collections import deque

_DEFAULT_RESPONSE = "mock response"


class MockLLM:
    """Deterministic LLM that cycles through a fixed list of responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._cycle: deque[str] = deque(responses if responses else [_DEFAULT_RESPONSE])

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        value = self._cycle[0]
        self._cycle.rotate(-1)
        return value


class ClaudeCliLLM:
    """LLM backend that calls the authenticated `claude` CLI."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        cmd = ["claude", "--model", self.model, "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    @staticmethod
    def available() -> bool:
        return shutil.which("claude") is not None
