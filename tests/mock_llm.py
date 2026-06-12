from __future__ import annotations

from collections import deque

from writing_assistant.types import LLMBackend


_DEFAULT_RESPONSE = "mock response"


class MockLLM:
    """Scripted LLM test double (lives in tests/ — never shipped).

    Pass a dict to key responses by exact prompt, or a list to cycle through
    responses in order (wrapping after the last entry).
    """

    def __init__(self, responses: dict[str, str] | list[str] | None = None) -> None:
        if isinstance(responses, list):
            self._dict: dict[str, str] | None = None
            self._cycle: deque[str] | None = deque(responses if responses else [_DEFAULT_RESPONSE])
        else:
            self._dict = responses or {}
            self._cycle = None

    def generate(self, prompt: str) -> str:
        if self._cycle is not None:
            value = self._cycle[0]
            self._cycle.rotate(-1)
            return value
        assert self._dict is not None
        return self._dict.get(prompt, _DEFAULT_RESPONSE)


# Satisfy the Protocol at class definition time
_: LLMBackend = MockLLM()
