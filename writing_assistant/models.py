from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMBackend(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class StyleProfile:
    """Captures target style constraints for a rewrite pass."""

    tone: str = "neutral"
    formality: str = "formal"
    max_sentence_length: int = 30
    preferred_voice: str = "active"
    custom_rules: list[str] = field(default_factory=list)


@dataclass
class RewritePass:
    """A single rewrite pass: input text, instructions, and which backend to use."""

    text: str
    instructions: str
    backend: LLMBackend = LLMBackend.CLAUDE
    style_profile: StyleProfile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassResult:
    """Output of a completed rewrite pass."""

    original: str
    rewritten: str
    pass_config: RewritePass
    backend_used: LLMBackend
    notes: str = ""
    success: bool = True
