from __future__ import annotations

from collections import deque
from typing import Literal

from writing_assistant.types import LLMBackend


_DEFAULT_RESPONSE = "mock response"


class MockLLM:
    """Scripted LLM test double (lives in tests/ — never shipped).

    Pass a dict to key responses by exact prompt, or a list to cycle through
    responses in order (wrapping after the last entry). ``fragment_match`` and
    the ``last-line`` fallback retain the recovered Human Writer test utility
    as explicit opt-ins; canonical exact and cycling behavior remain defaults.
    """

    def __init__(
        self,
        responses: dict[str, str] | list[str] | None = None,
        *,
        fragment_match: bool = False,
        fallback: Literal["default", "last-line"] = "default",
    ) -> None:
        if fallback not in {"default", "last-line"}:
            raise ValueError(f"Unknown MockLLM fallback: {fallback!r}")
        if isinstance(responses, list):
            if fragment_match or fallback != "default":
                raise ValueError("fragment matching and fallback selection require dict mode")
            self._dict: dict[str, str] | None = None
            self._cycle: deque[str] | None = deque(responses if responses else [_DEFAULT_RESPONSE])
        else:
            self._dict = responses or {}
            self._cycle = None
        self._fragment_match = fragment_match
        self._fallback = fallback

    def generate(self, prompt: str) -> str:
        if self._cycle is not None:
            value = self._cycle[0]
            self._cycle.rotate(-1)
            return value
        assert self._dict is not None
        if self._fragment_match:
            for key, response in self._dict.items():
                if key in prompt:
                    return response
        elif prompt in self._dict:
            return self._dict[prompt]
        if self._fallback == "last-line":
            stripped = prompt.strip()
            last_line = stripped.splitlines()[-1] if stripped else ""
            return f"[mock] {last_line}"
        return _DEFAULT_RESPONSE


# Satisfy the Protocol at class definition time
_: LLMBackend = MockLLM()
