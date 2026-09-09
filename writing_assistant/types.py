from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from writing_assistant.style import DesiredStyleProfile, StyleProfile


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class Pass:
    name: str
    instructions: str
    metadata: dict = field(default_factory=dict)
    executor: Callable[[str, StyleProfile | DesiredStyleProfile | None, LLMBackend], str] | None = None

    def __post_init__(self) -> None:
        if self.executor is not None and not callable(self.executor):
            raise TypeError("executor must be callable or None")


@dataclass
class RewriteResult:
    original: str
    revised: str
    diff: str
    pass_name: str | None = None
