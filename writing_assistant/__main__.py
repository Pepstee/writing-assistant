"""CLI entry point: python -m writing_assistant [options] [file]"""

from __future__ import annotations

import argparse
import sys

from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.style import StyleProfile

_PASS_MAP = {
    "clarity": CLARITY,
    "tone": TONE,
    "conciseness": CONCISENESS,
    "consistency": CONSISTENCY,
    "adversarial": ADVERSARIAL,
}

_DEFAULT_PASSES = ["clarity", "tone", "conciseness", "consistency", "adversarial"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m writing_assistant",
        description="Multi-pass writing assistant pipeline.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="text file to process (omit to read from stdin)",
    )
    parser.add_argument(
        "--sample",
        metavar="FILE",
        help="text file to learn style profile from (used by the consistency pass)",
    )
    parser.add_argument(
        "--passes",
        default=",".join(_DEFAULT_PASSES),
        metavar="LIST",
        help=(
            f"comma-separated passes to run "
            f"(default: {','.join(_DEFAULT_PASSES)}; "
            f"available: {','.join(_PASS_MAP)})"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the mock LLM backend instead of the Claude CLI",
    )
    args = parser.parse_args()

    # Read input text
    if args.input:
        try:
            with open(args.input) as fh:
                text = fh.read()
        except OSError as exc:
            parser.error(str(exc))
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        parser.error("input text is empty")

    # Resolve pass list
    pass_names = [p.strip() for p in args.passes.split(",") if p.strip()]
    unknown = [p for p in pass_names if p not in _PASS_MAP]
    if unknown:
        parser.error(
            f"unknown pass(es): {', '.join(unknown)}. "
            f"Available: {', '.join(_PASS_MAP)}"
        )
    passes = [_PASS_MAP[n] for n in pass_names]

    # Optional style profile
    style_profile: StyleProfile | None = None
    if args.sample:
        try:
            with open(args.sample) as fh:
                sample_text = fh.read()
        except OSError as exc:
            parser.error(str(exc))
        style_profile = StyleProfile.learn([sample_text])
        print(f"Style profile learned from {args.sample!r}:")
        print(style_profile.summary())
        print()

    # Build backend
    if args.mock:
        from writing_assistant.llm.mock import MockLLM

        backend: object = MockLLM()
    else:
        from writing_assistant.backends import ClaudeCLIBackend

        backend = ClaudeCLIBackend()

    # Run
    pipeline = Pipeline(passes=passes, backend=backend, style_profile=style_profile)
    results = pipeline.run(text)

    # Structured output
    for p, result in zip(passes, results):
        print(f"\n{'=' * 60}")
        print(f"Pass: {p.name.upper()}")
        print("=" * 60)
        if p.metadata.get("adversarial"):
            print("Adversarial review: identifying weaknesses and rewriting to address them.")
        if result.diff:
            print(result.diff)
        else:
            print("(no changes)")

    print(f"\n{'=' * 60}")
    print("Final draft:")
    print("=" * 60)
    print(results[-1].revised)


if __name__ == "__main__":
    main()
