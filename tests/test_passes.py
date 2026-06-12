"""Adversarial tests for writing_assistant.passes.

Mutation-resistant: every test fails if a Pass instance is removed, renamed,
loses its instructions, gains/loses the adversarial metadata flag, or two
passes collapse into sharing the same instructions.
"""
from __future__ import annotations

import pytest

from writing_assistant import passes as passes_module
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.types import Pass

ALL_PASSES = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]

EXPECTED_NAMES = {
    "CLARITY": "clarity",
    "TONE": "tone",
    "CONCISENESS": "conciseness",
    "CONSISTENCY": "consistency",
    "ADVERSARIAL": "adversarial",
}


class TestAllFivePassesExist:
    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_pass_constant_exists(self, attr):
        assert hasattr(passes_module, attr), f"passes.{attr} is missing"

    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_pass_constant_is_pass_instance(self, attr):
        value = getattr(passes_module, attr)
        assert isinstance(value, Pass), f"passes.{attr} is {type(value)}, expected Pass"

    def test_exactly_five_passes(self):
        assert len(ALL_PASSES) == 5

    def test_module_exposes_no_extra_pass_instances(self):
        exported = {
            name for name in vars(passes_module)
            if isinstance(getattr(passes_module, name), Pass)
        }
        assert exported == set(EXPECTED_NAMES), (
            f"Unexpected Pass exports: {sorted(exported ^ set(EXPECTED_NAMES))}"
        )


class TestPassNames:
    @pytest.mark.parametrize("attr,name", sorted(EXPECTED_NAMES.items()))
    def test_name_matches_constant(self, attr, name):
        assert getattr(passes_module, attr).name == name

    def test_all_names_distinct(self):
        names = [p.name for p in ALL_PASSES]
        assert len(set(names)) == len(names), f"Duplicate pass names: {names}"

    def test_names_are_lowercase_identifiers(self):
        for p in ALL_PASSES:
            assert p.name == p.name.lower()
            assert p.name.isidentifier()


class TestPassInstructions:
    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_instructions_non_empty(self, attr):
        instructions = getattr(passes_module, attr).instructions
        assert isinstance(instructions, str)
        assert instructions.strip(), f"passes.{attr}.instructions is empty"

    def test_all_instructions_distinct(self):
        texts = [p.instructions for p in ALL_PASSES]
        assert len(set(texts)) == len(texts), "Two passes share identical instructions"

    def test_instructions_are_substantive(self):
        # A one-word instruction is a stub; require a real sentence.
        for p in ALL_PASSES:
            assert len(p.instructions.split()) >= 8, (
                f"{p.name} instructions look like a stub: {p.instructions!r}"
            )

    def test_clarity_instructions_mention_clarity(self):
        assert "clarity" in CLARITY.instructions.lower()

    def test_tone_instructions_mention_tone(self):
        assert "tone" in TONE.instructions.lower()

    def test_conciseness_instructions_mention_concise(self):
        assert "concise" in CONCISENESS.instructions.lower()

    def test_consistency_instructions_mention_consistent(self):
        assert "consisten" in CONSISTENCY.instructions.lower()

    def test_adversarial_instructions_mention_weaknesses(self):
        text = ADVERSARIAL.instructions.lower()
        assert "adversarial" in text
        assert "weakness" in text


class TestAdversarialMetadataFlag:
    def test_adversarial_pass_flagged(self):
        assert ADVERSARIAL.metadata.get("adversarial") is True

    @pytest.mark.parametrize(
        "non_adversarial", [CLARITY, TONE, CONCISENESS, CONSISTENCY],
        ids=lambda p: p.name,
    )
    def test_only_adversarial_carries_flag(self, non_adversarial):
        assert not non_adversarial.metadata.get("adversarial"), (
            f"{non_adversarial.name} must not carry the adversarial flag"
        )

    def test_metadata_is_dict(self):
        for p in ALL_PASSES:
            assert isinstance(p.metadata, dict)

    def test_metadata_not_shared_between_instances(self):
        # A shared default dict would let one pass mutate another's metadata.
        CLARITY.metadata["__probe__"] = 1
        try:
            assert "__probe__" not in TONE.metadata
            assert "__probe__" not in ADVERSARIAL.metadata
        finally:
            CLARITY.metadata.pop("__probe__", None)


class TestLegacyApiIsGone:
    """The class-based pass API was removed; resurrecting it must fail loudly."""

    @pytest.mark.parametrize("legacy_name", [
        "ClarityPass", "TonePass", "ConcisenessPass", "ConsistencyPass",
        "AdversarialPass", "_BasePass", "PassResult", "_unified_diff",
    ])
    def test_legacy_symbol_absent(self, legacy_name):
        assert not hasattr(passes_module, legacy_name), (
            f"Legacy symbol passes.{legacy_name} has been resurrected"
        )
