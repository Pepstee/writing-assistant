from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from writer.passes import DEFAULT_PASSES, Pass


@dataclass
class PassResult:
    pass_name: str
    original: str
    revised: str
    diff: str


@dataclass
class PipelineResult:
    pass_results: list[PassResult] = field(default_factory=list)
    final_text: str = ""

    def __post_init__(self) -> None:
        if not self.final_text and self.pass_results:
            self.final_text = self.pass_results[-1].revised


class RewritePipeline:
    """Multi-pass rewrite pipeline."""

    def __init__(self, llm: object, passes: list[Pass] | None = None) -> None:
        self.llm = llm
        self.passes = passes if passes is not None else DEFAULT_PASSES

    def run(self, text: str) -> PipelineResult:
        results: list[PassResult] = []
        current = text
        for p in self.passes:
            prompt = f"{p.instructions}\n\nText:\n{current}"
            revised = self.llm.generate(prompt)  # type: ignore[union-attr]
            diff = "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    revised.splitlines(keepends=True),
                    fromfile="input",
                    tofile="revised",
                )
            )
            results.append(
                PassResult(pass_name=p.name, original=current, revised=revised, diff=diff)
            )
            current = revised
        return PipelineResult(pass_results=results, final_text=current)
