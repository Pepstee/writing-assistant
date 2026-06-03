from __future__ import annotations

from writing_assistant.types import LLMBackend


_DEFAULT_RESPONSE = "mock response"


class MockLLM:
    """Deterministic LLM backend that returns prompt-keyed responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses: dict[str, str] = responses or {}

    def generate(self, prompt: str) -> str:
        return self._responses.get(prompt, _DEFAULT_RESPONSE)


# Satisfy the Protocol at class definition time (structural check via runtime isinstance is optional)
_: LLMBackend = MockLLM()
