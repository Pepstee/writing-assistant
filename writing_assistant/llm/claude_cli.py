from __future__ import annotations

import subprocess

from writing_assistant.types import LLMBackend


class ClaudeCliLLM:
    """LLM backend that shells out to the authenticated `claude` CLI."""

    def __init__(
        self, model: str = "claude-sonnet-4-6", extra_args: list[str] | None = None
    ) -> None:
        self.model = model
        self.extra_args = extra_args or []

    def generate(self, prompt: str) -> str:
        cmd = ["claude", "--model", self.model, *self.extra_args, "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()


_: LLMBackend = ClaudeCliLLM()
