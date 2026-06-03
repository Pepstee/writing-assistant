#!/usr/bin/env python3
"""Writing Assistant demo — uses Claude CLI if available, falls back to MockLLM."""

from writer.llm import ClaudeCliLLM, MockLLM
from writer.passes import ADVERSARIAL, CLARITY, CONCISENESS, TONE
from writer.pipeline import RewritePipeline

SAMPLE = (
    "In the event that you are considering making utilization of our software product, "
    "it is of the utmost importance to take into consideration the fact that there are "
    "numerous features which have been designed in order to facilitate the process of "
    "improving your written content. The aforementioned features include clarity "
    "enhancement, conciseness improvements, and tone adjustments for professional audiences."
)

MOCK_RESPONSES = [
    (
        "If you're considering our software, know that it offers features to improve your writing: "
        "clarity enhancement, conciseness improvements, and tone adjustment."
    ),
    (
        "Our software improves writing through three features: clarity enhancement, "
        "conciseness, and tone adjustment."
    ),
    (
        "Our software improves writing with three focused features: clarity, conciseness, "
        "and tone adjustment — suitable for professional audiences."
    ),
    (
        "Our software improves writing with three focused features: clarity, conciseness, "
        "and tone adjustment — suitable for professional audiences."
    ),
]

PASSES = [CLARITY, CONCISENESS, TONE, ADVERSARIAL]


def main() -> None:
    if ClaudeCliLLM.available():
        llm: object = ClaudeCliLLM()
        backend_name = "Claude CLI"
    else:
        llm = MockLLM(responses=MOCK_RESPONSES)
        backend_name = "MockLLM (Claude CLI unavailable)"

    pipeline = RewritePipeline(llm=llm, passes=PASSES)

    print("=" * 60)
    print("Writing Assistant — Demo")
    print(f"Backend: {backend_name}")
    print("=" * 60)
    print(f"\nOriginal draft ({len(SAMPLE.split())} words):")
    print(SAMPLE)

    result = pipeline.run(SAMPLE)

    for pr in result.pass_results:
        print(f"\n--- Pass: {pr.pass_name} ---")
        if pr.diff:
            print("Diff:")
            print(pr.diff)
        else:
            print("(no changes from this pass)")

    print("\n" + "=" * 60)
    print("Final rewrite:")
    print(result.final_text)
    print("=" * 60)
    print(f"\nStyle: professional | Passes run: {len(result.pass_results)}")


if __name__ == "__main__":
    main()
