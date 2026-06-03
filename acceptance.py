#!/usr/bin/env python3
"""Acceptance demo: runs the full pipeline on a bundled sample draft using MockBackend."""

from writing_assistant.backends import MockBackend
from writing_assistant.pipeline import Pipeline, PipelineConfig

SAMPLE_DRAFT = (
    "In the event that you are considering making utilization of our software product, "
    "it is important to take into consideration the fact that there are numerous "
    "features which have been implemented in order to facilitate the process of "
    "improving your written content. The aforementioned features include, but are "
    "not limited to, clarity enhancement, conciseness improvements, and adversarial "
    "self-review capabilities which are available for your use at any point in time."
)

# One deterministic response per pass; MockBackend cycles through these.
MOCK_RESPONSES = [
    # clarity pass
    (
        "If you are considering using our software, note that it offers many features "
        "to help improve your written content. These include clarity enhancement, "
        "conciseness improvements, and adversarial self-review, all available at any time."
    ),
    # conciseness pass
    (
        "Our software offers features to improve your writing: "
        "clarity enhancement, conciseness improvements, and adversarial self-review."
    ),
    # adversarial pass
    (
        "Our software improves your writing through three focused features: "
        "clarity enhancement, conciseness, and adversarial self-review."
    ),
]

PASSES = ["clarity", "conciseness", "adversarial"]


def main() -> None:
    backend = MockBackend(responses=MOCK_RESPONSES)
    config = PipelineConfig(passes=PASSES)
    pipeline = Pipeline(config=config, backend=backend)

    print("=" * 60)
    print("Writing Assistant — Acceptance Demo")
    print("=" * 60)
    print(f"\nOriginal draft ({len(SAMPLE_DRAFT)} chars):")
    print(SAMPLE_DRAFT)

    result = pipeline.run(SAMPLE_DRAFT)

    for i, (pass_name, pass_result) in enumerate(zip(PASSES, result.pass_results), 1):
        print(f"\n--- Pass {i}: {pass_name} ---")
        if pass_result.diff:
            print("Diff:")
            print(pass_result.diff)
        else:
            print("Diff: (no changes)")

    print("\n" + "=" * 60)
    print("Final rewrite:")
    print(result.final_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
