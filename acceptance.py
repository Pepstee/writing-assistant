#!/usr/bin/env python3
"""Acceptance demo: runs the full pipeline on a 150-word sample draft using MockLLM."""

from writing_assistant.llm.mock import MockLLM
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline

SAMPLE_DRAFT = (
    "In the event that you are considering making utilization of our software product, "
    "it is of the utmost importance to take into consideration the fact that there are "
    "numerous features which have been implemented in order to facilitate the process of "
    "improving your written content in a meaningful way. The aforementioned features include, "
    "but are not limited to, clarity enhancement capabilities, conciseness improvements that "
    "reduce verbosity, tone adjustments for professional audiences, consistency enforcement "
    "across the entirety of your document, and adversarial self-review capabilities which "
    "are available for your use at any point in time. It should furthermore be noted that "
    "the system has been designed with the explicit purpose of ensuring that all users, "
    "regardless of their level of expertise or experience, are able to derive maximum benefit "
    "from the comprehensive suite of writing tools that has been made available to them."
)

MOCK_RESPONSES = [
    # clarity
    (
        "If you are considering using our software, know that it offers many features "
        "to improve your writing. These include clarity enhancement, conciseness improvements, "
        "tone adjustment, consistency enforcement, and adversarial self-review, all available "
        "at any time. The system is designed so that users of all skill levels can benefit "
        "from its comprehensive writing tools."
    ),
    # tone
    (
        "If you are considering our software, it provides several features to improve your "
        "writing: clarity enhancement, conciseness improvements, tone adjustment, consistency "
        "enforcement, and adversarial self-review. These tools are accessible to users at all "
        "experience levels."
    ),
    # conciseness
    (
        "Our software improves your writing through five focused features: clarity enhancement, "
        "conciseness, tone adjustment, consistency enforcement, and adversarial self-review — "
        "all available to users at every skill level."
    ),
    # consistency
    (
        "Our software improves writing through five features: clarity enhancement, conciseness "
        "improvement, tone adjustment, consistency enforcement, and adversarial self-review — "
        "available to writers at every skill level."
    ),
    # adversarial — no regression found; returns the same text
    (
        "Our software improves writing through five features: clarity enhancement, conciseness "
        "improvement, tone adjustment, consistency enforcement, and adversarial self-review — "
        "available to writers at every skill level."
    ),
]

PASSES = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]


def main() -> None:
    backend = MockLLM(responses=MOCK_RESPONSES)
    pipeline = Pipeline(passes=PASSES, backend=backend)

    print("=" * 60)
    print("Writing Assistant — Acceptance Demo")
    print("=" * 60)
    print(f"\nOriginal draft ({len(SAMPLE_DRAFT.split())} words):")
    print(SAMPLE_DRAFT)

    results = pipeline.run(SAMPLE_DRAFT)

    for p, result in zip(PASSES, results):
        print(f"\n--- Pass: {p.name} ---")
        if result.diff:
            print("Diff:")
            print(result.diff)
        else:
            print("Diff: (no changes — adversarial pass found no regression)")

    print("\n" + "=" * 60)
    print("Final rewrite:")
    print(results[-1].revised)
    print("=" * 60)


if __name__ == "__main__":
    main()
