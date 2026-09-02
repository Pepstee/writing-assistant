"""The five built-in rewrite passes.

Each pass is a plain :class:`~writing_assistant.types.Pass` dataclass instance;
the pipeline turns its ``instructions`` into the LLM prompt. The adversarial
pass is marked via ``metadata`` so the pipeline gives it the full rewrite
history to critique.
"""
from __future__ import annotations

from writing_assistant.types import Pass


class PassRegistry:
    """Validated name-to-pass registry for pipeline and CLI composition.

    Canonical passes are data objects rather than donor-style ``RewritePass``
    subclasses, so this registry stores and returns the exact :class:`Pass`
    instance registered by its caller.
    """

    def __init__(self) -> None:
        self._passes: dict[str, Pass] = {}

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"name must be a non-empty string, got {name!r}")

    def register(self, name: str, rewrite_pass: Pass) -> None:
        """Register *rewrite_pass* under *name* without replacing an owner."""

        self._validate_name(name)
        if not isinstance(rewrite_pass, Pass):
            raise TypeError(f"{rewrite_pass!r} is not a Pass instance")
        if name in self._passes:
            raise ValueError(f"a pass named {name!r} is already registered")
        self._passes[name] = rewrite_pass

    def get(self, name: str) -> Pass:
        """Return the exact pass registered under *name*."""

        self._validate_name(name)
        try:
            return self._passes[name]
        except KeyError as exc:
            raise KeyError(
                f"no pass registered under {name!r}; available passes: {self.names()}"
            ) from exc

    def names(self) -> list[str]:
        """Return registered names in deterministic alphabetical order."""

        return sorted(self._passes)

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


BUILTIN_PASS_REGISTRY = PassRegistry()
for _pass in (CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL):
    BUILTIN_PASS_REGISTRY.register(_pass.name, _pass)
del _pass
