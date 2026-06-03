from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class Pass:
    name: str
    instructions: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RewriteResult:
    original: str
    revised: str
    diff: str
