from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any


@dataclass
class PassResult:
    original: str
    rewritten: str
    diff: str
    notes: str = ""
    success: bool = True


def _unified_diff(original: str, rewritten: str) -> str:
    a = original.splitlines(keepends=True)
    b = rewritten.splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile="original", tofile="rewritten"))


class _BasePass:
    INSTRUCTIONS: str = ""

    def run(self, text: str, backend: Any) -> PassResult:
        prompt = f"{self.INSTRUCTIONS}\n\nText:\n{text}"
        rewritten = backend.complete(prompt)
        return PassResult(
            original=text,
            rewritten=rewritten,
            diff=_unified_diff(text, rewritten),
        )


class ClarityPass(_BasePass):
    INSTRUCTIONS = (
        "Rewrite the following text to improve clarity. "
        "Use plain language, avoid jargon, and make every sentence easy to understand."
    )


class TonePass(_BasePass):
    INSTRUCTIONS = (
        "Rewrite the following text to achieve a professional, respectful tone "
        "appropriate for a general audience."
    )


class ConcisenessPass(_BasePass):
    INSTRUCTIONS = (
        "Rewrite the following text to be more concise. "
        "Remove redundant words, filler phrases, and unnecessary detail while preserving all meaning."
    )


class ConsistencyPass(_BasePass):
    INSTRUCTIONS = (
        "Rewrite the following text to ensure consistent terminology, voice, and style throughout."
    )


class AdversarialPass(_BasePass):
    """Self-review pass: surfaces weaknesses then rewrites to address them."""

    INSTRUCTIONS = (
        "Act as an adversarial editor. First identify weaknesses: unclear claims, logical gaps, "
        "redundancy, and stylistic flaws. Then produce an improved version that addresses each "
        "identified weakness. Return only the improved text."
    )


from writing_assistant.types import Pass  # noqa: E402

CLARITY = Pass(name="clarity", instructions=ClarityPass.INSTRUCTIONS)
TONE = Pass(name="tone", instructions=TonePass.INSTRUCTIONS)
CONCISENESS = Pass(name="conciseness", instructions=ConcisenessPass.INSTRUCTIONS)
CONSISTENCY = Pass(name="consistency", instructions=ConsistencyPass.INSTRUCTIONS)
ADVERSARIAL = Pass(
    name="adversarial",
    instructions=AdversarialPass.INSTRUCTIONS,
    metadata={"adversarial": True},
)
