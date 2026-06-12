"""The five built-in rewrite passes.

Each pass is a plain :class:`~writing_assistant.types.Pass` dataclass instance;
the pipeline turns its ``instructions`` into the LLM prompt. The adversarial
pass is marked via ``metadata`` so the pipeline gives it the full rewrite
history to critique.
"""
from __future__ import annotations

from writing_assistant.types import Pass

CLARITY = Pass(
    name="clarity",
    instructions=(
        "Rewrite the following text to improve clarity. "
        "Use plain language, avoid jargon, and make every sentence easy to understand."
    ),
)

TONE = Pass(
    name="tone",
    instructions=(
        "Rewrite the following text to achieve a professional, respectful tone "
        "appropriate for a general audience."
    ),
)

CONCISENESS = Pass(
    name="conciseness",
    instructions=(
        "Rewrite the following text to be more concise. "
        "Remove redundant words, filler phrases, and unnecessary detail "
        "while preserving all meaning."
    ),
)

CONSISTENCY = Pass(
    name="consistency",
    instructions=(
        "Rewrite the following text to ensure consistent terminology, voice, and style throughout."
    ),
)

ADVERSARIAL = Pass(
    name="adversarial",
    instructions=(
        "Act as an adversarial editor. First identify weaknesses: unclear claims, logical gaps, "
        "redundancy, and stylistic flaws. Then produce an improved version that addresses each "
        "identified weakness. Return only the improved text."
    ),
    metadata={"adversarial": True},
)
