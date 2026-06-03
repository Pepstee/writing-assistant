from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pass:
    name: str
    instructions: str
    metadata: dict = field(default_factory=dict)


CLARITY = Pass(
    name="clarity",
    instructions=(
        "Rewrite the following text to improve clarity. "
        "Use plain language, avoid jargon, and make every sentence easy to understand."
    ),
)

CONCISENESS = Pass(
    name="conciseness",
    instructions=(
        "Rewrite the following text to be more concise. "
        "Remove redundant words and filler phrases while preserving all meaning."
    ),
)

TONE = Pass(
    name="tone",
    instructions=(
        "Rewrite the following text to achieve a professional, respectful tone "
        "appropriate for a general audience."
    ),
)

ADVERSARIAL = Pass(
    name="adversarial",
    instructions=(
        "Act as an adversarial editor. Identify weaknesses — unclear claims, logical gaps, "
        "redundancy, and stylistic flaws — then produce an improved version that addresses each. "
        "Return only the improved text."
    ),
    metadata={"adversarial": True},
)

DEFAULT_PASSES: list[Pass] = [CLARITY, CONCISENESS, TONE]
