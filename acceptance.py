#!/usr/bin/env python3
"""Acceptance demo: the full product, end to end, offline.

Runs the real five-pass pipeline (clarity, tone, conciseness, consistency,
adversarial self-review) over a deliberately wordy 150-word draft using the
deterministic rule-based backend — every edit shown is a genuine
transformation made by shipped code, with no network and no scripted
responses. Also demonstrates the style-profile system by learning a profile
from a sample text and wiring it into the consistency pass.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from writing_assistant.llm.rule_based import RuleBasedRewriter
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.style import StyleProfile

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

STYLE_SAMPLE = (
    "We build tools that respect the reader. Every sentence earns its place. "
    "However, brevity never excuses vagueness; therefore, each claim is concrete. "
    "Furthermore, we prefer the active voice and plain words."
)

PASSES = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]


def main() -> None:
    profile = StyleProfile.learn([STYLE_SAMPLE])
    backend = RuleBasedRewriter()
    pipeline = Pipeline(passes=PASSES, backend=backend, style_profile=profile)

    print("=" * 60)
    print("Writing Assistant — Acceptance Demo (offline, deterministic)")
    print("=" * 60)

    print("\nStyle profile learned from sample:")
    print(profile.summary())

    print(f"\nOriginal draft ({len(SAMPLE_DRAFT.split())} words):")
    print(SAMPLE_DRAFT)

    results = pipeline.run(SAMPLE_DRAFT)

    for p, result in zip(PASSES, results):
        print(f"\n--- Pass: {p.name} ---")
        if result.diff:
            print("Diff:")
            print(result.diff)
        elif p.metadata.get("adversarial"):
            print("Diff: (no changes — adversarial self-review found no further weaknesses)")
        else:
            print(f"Diff: (no changes — the {p.name} pass found nothing to edit)")

    final = results[-1].revised
    print("\n" + "=" * 60)
    print(f"Final rewrite: ({len(final.split())} words, "
          f"down from {len(SAMPLE_DRAFT.split())})")
    print(final)
    print("=" * 60)

    # The demo must demonstrate, not merely run: fail loudly if the pipeline
    # did not actually tighten the draft.
    assert len(final.split()) < len(SAMPLE_DRAFT.split()), (
        "pipeline failed to shorten the wordy draft"
    )
    assert "in the event that" not in final.lower(), (
        "clarity pass failed to remove wordy phrasing"
    )


if __name__ == "__main__":
    main()
