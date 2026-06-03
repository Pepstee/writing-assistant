from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from writing_assistant.models import StyleProfile as _LegacyStyleProfile
from writing_assistant.style import StyleProfile
from writing_assistant.passes import (
    AdversarialPass,
    ClarityPass,
    ConcisenessPass,
    ConsistencyPass,
    PassResult,
    TonePass,
)
from writing_assistant.types import LLMBackend, Pass, RewriteResult

_PASS_REGISTRY: dict[str, Any] = {
    "clarity": ClarityPass,
    "conciseness": ConcisenessPass,
    "tone": TonePass,
    "consistency": ConsistencyPass,
    "adversarial": AdversarialPass,
}


@dataclass
class PipelineConfig:
    passes: list[str]
    style_profile: _LegacyStyleProfile | None = None


@dataclass
class PipelineResult:
    pass_results: list[PassResult]
    final_text: str = ""

    def __post_init__(self) -> None:
        if not self.final_text and self.pass_results:
            self.final_text = self.pass_results[-1].rewritten


class LegacyPipeline:
    def __init__(self, config: PipelineConfig, backend: Any) -> None:
        self.config = config
        self.backend = backend
        self._passes = [_PASS_REGISTRY[name]() for name in config.passes]

    def run(self, text: str) -> PipelineResult:
        results: list[PassResult] = []
        current = text
        for pass_obj in self._passes:
            result = pass_obj.run(current, self.backend)
            results.append(result)
            current = result.rewritten
        return PipelineResult(pass_results=results, final_text=current)


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
        for i, p in enumerate(self.passes):
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
