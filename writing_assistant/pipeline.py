from __future__ import annotations

import difflib

from writing_assistant.style import StyleProfile
from writing_assistant.types import LLMBackend, Pass, RewriteResult


class Pipeline:
    """Multi-pass rewrite pipeline accepting Pass dataclass instances."""

    def __init__(
        self,
        passes: list[Pass],
        backend: LLMBackend,
        style_profile: StyleProfile | None = None,
    ) -> None:
        self.passes = passes
        self.backend = backend
        self.style_profile = style_profile

    def run(self, text: str) -> list[RewriteResult]:
        results: list[RewriteResult] = []
        current = text
        for p in self.passes:
            if not p.instructions.strip():
                raise ValueError(
                    f"Pass '{p.name}' has empty or whitespace-only instructions."
                )
            if p.metadata.get("adversarial"):
                history = "\n\n".join(
                    f"[{self.passes[j].name}]\n{results[j].revised}"
                    for j in range(len(results))
                )
                prompt = (
                    f"{p.instructions}\n\n"
                    f"Original text:\n{text}\n\n"
                    f"Accumulated rewrites:\n{history}\n\n"
                    f"Current text:\n{current}"
                )
            else:
                instructions = p.instructions
                if p.name == "consistency" and self.style_profile is not None:
                    instructions = (
                        f"Style profile:\n{self.style_profile.summary()}\n\n{instructions}"
                    )
                prompt = f"{instructions}\n\nText:\n{current}"
            revised = self.backend.generate(prompt)
            diff = "".join(difflib.unified_diff(
                current.splitlines(keepends=True),
                revised.splitlines(keepends=True),
                fromfile="input",
                tofile="revised",
            ))
            results.append(RewriteResult(original=current, revised=revised, diff=diff))
            current = revised
        return results
