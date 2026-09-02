from __future__ import annotations

import difflib

from writing_assistant.style import DesiredStyleProfile, StyleProfile
from writing_assistant.types import LLMBackend, Pass, RewriteResult


_BUILT_IN_PASS_NAMES = frozenset({"clarity", "tone", "conciseness", "consistency", "adversarial"})


def _unified_diff(original: str, revised: str) -> str:
    """Return a readable diff without erasing exact line-ending changes."""

    chunks: list[str] = []
    for line in difflib.unified_diff(
        original.splitlines(keepends=True),
        revised.splitlines(keepends=True),
        fromfile="input",
        tofile="revised",
    ):
        chunks.append(line)
        if line[:1] in {" ", "+", "-"} and not line.endswith(("\n", "\r")):
            chunks.append("\n\\ No newline at end of file\n")
    return "".join(chunks)


class Pipeline:
    """Multi-pass rewrite pipeline accepting Pass dataclass instances."""

    def __init__(
        self,
        passes: list[Pass],
        backend: LLMBackend,
        style_profile: StyleProfile | DesiredStyleProfile | None = None,
    ) -> None:
        self.passes = passes
        self.backend = backend
        self.style_profile = style_profile

    def run(self, text: str) -> list[RewriteResult]:
        results: list[RewriteResult] = []
        current = text
        for p in self.passes:
            if not p.instructions.strip():
                raise ValueError(f"Pass '{p.name}' has empty or whitespace-only instructions.")
            guidance: str | None = None
            if isinstance(self.style_profile, DesiredStyleProfile):
                guidance = f"Desired style:\n{self.style_profile.summary()}"
            elif p.name == "consistency" and self.style_profile is not None:
                guidance = f"Style profile:\n{self.style_profile.summary()}"
            guidance_prefix = f"{guidance}\n\n" if guidance else ""
            pass_identity = p.name if p.name in _BUILT_IN_PASS_NAMES else "custom"
            prompt_prefix = f"Rewrite pass: {pass_identity}\nPayload characters: {len(current)}\n\n"
            instruction_block = f"Pass instructions:\n{p.instructions}"

            if p.metadata.get("adversarial"):
                history = "\n\n".join(
                    f"[{self.passes[j].name}]\n{results[j].revised}" for j in range(len(results))
                )
                prompt = (
                    f"{prompt_prefix}{guidance_prefix}{instruction_block}\n\n"
                    f"Original text:\n{text}\n\n"
                    f"Accumulated rewrites:\n{history}\n\n"
                    f"Payload:\n{current}"
                )
            else:
                instructions = f"{guidance_prefix}{instruction_block}"
                prompt = f"{prompt_prefix}{instructions}\n\nPayload:\n{current}"
            revised = self.backend.generate(prompt)
            diff = _unified_diff(current, revised)
            results.append(
                RewriteResult(
                    original=current,
                    revised=revised,
                    diff=diff,
                    pass_name=p.name,
                )
            )
            current = revised
        return results
