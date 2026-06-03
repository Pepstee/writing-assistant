from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from writing_assistant.models import StyleProfile
from writing_assistant.passes import (
    AdversarialPass,
    ClarityPass,
    ConcisenessPass,
    ConsistencyPass,
    PassResult,
    TonePass,
)

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
    style_profile: StyleProfile | None = None


@dataclass
class PipelineResult:
    pass_results: list[PassResult]
    final_text: str = ""

    def __post_init__(self) -> None:
        if not self.final_text and self.pass_results:
            self.final_text = self.pass_results[-1].rewritten


class Pipeline:
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
