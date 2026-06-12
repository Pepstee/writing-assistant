"""Adversarial tests for writing_assistant.passes.

Mutation-resistant: every test fails if a Pass instance is removed, renamed,
loses its instructions, gains/loses the adversarial metadata flag, or two
passes collapse into sharing the same instructions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mock_llm import MockLLM
from writing_assistant import passes as passes_module
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.types import Pass

_REPO_ROOT = Path(__file__).parent.parent

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


class TestPassInstructionsMutationBlocksPipeline:
    """Adversarial: zeroing a pass's instructions must cause the pipeline to
    raise (ValueError) rather than silently generating prompt-free output.

    Each test restores the original instructions in a finally block so that
    module-level constants are not permanently corrupted across the suite.
    """

    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_empty_instructions_raises_in_pipeline(self, attr):
        p = getattr(passes_module, attr)
        original = p.instructions
        try:
            p.instructions = ""
            with pytest.raises(ValueError):
                Pipeline(passes=[p], backend=MockLLM(["output"])).run("Some input text.")
        finally:
            p.instructions = original

    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_whitespace_only_instructions_raises_in_pipeline(self, attr):
        p = getattr(passes_module, attr)
        original = p.instructions
        try:
            p.instructions = "   \t\n"
            with pytest.raises(ValueError):
                Pipeline(passes=[p], backend=MockLLM(["output"])).run("Some input text.")
        finally:
            p.instructions = original

    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_instructions_restored_after_mutation(self, attr):
        """Finaliser must restore the constant even when the pipeline does not raise."""
        p = getattr(passes_module, attr)
        original = p.instructions
        try:
            p.instructions = ""
            try:
                Pipeline(passes=[p], backend=MockLLM(["output"])).run("Text.")
            except Exception:
                pass
        finally:
            p.instructions = original
        assert p.instructions == original, (
            f"passes.{attr}.instructions was not restored after mutation; "
            "module-level constant is corrupted"
        )

    def test_new_pass_with_empty_instructions_raises_in_pipeline(self):
        """A freshly constructed Pass with empty instructions must also raise."""
        p = Pass(name="empty_test", instructions="", metadata={})
        with pytest.raises(ValueError):
            Pipeline(passes=[p], backend=MockLLM(["output"])).run("Text.")

    def test_new_pass_with_whitespace_instructions_raises_in_pipeline(self):
        """A freshly constructed Pass with whitespace-only instructions must also raise."""
        p = Pass(name="ws_test", instructions="\n  \t  \n", metadata={})
        with pytest.raises(ValueError):
            Pipeline(passes=[p], backend=MockLLM(["output"])).run("Text.")

    @pytest.mark.parametrize("attr", sorted(EXPECTED_NAMES))
    def test_valid_instructions_still_work_after_mutation_cycle(self, attr):
        """Restoring original instructions must leave the pass fully functional."""
        p = getattr(passes_module, attr)
        original = p.instructions
        try:
            p.instructions = ""
        finally:
            p.instructions = original
        # Pass must now work without raising
        results = Pipeline(passes=[p], backend=MockLLM(["clean output."])).run("Text.")
        assert len(results) == 1


class TestAcceptanceDemoReliability:
    """acceptance.py must run to completion and exit 0 using the deterministic
    offline backend (no network calls, no API keys) and must satisfy its own
    built-in assertions: the pipeline shortens the draft and removes the
    canonical wordy phrase 'in the event that'."""

    _SCRIPT = _REPO_ROOT / "acceptance.py"

    def test_acceptance_demo_subprocess_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(self._SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"acceptance.py exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_acceptance_demo_subprocess_is_idempotent(self):
        """Running acceptance.py twice must produce exit 0 both times."""
        for run in range(2):
            result = subprocess.run(
                [sys.executable, str(self._SCRIPT)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"acceptance.py exited {result.returncode} on run {run + 1}.\n"
                f"stderr: {result.stderr}"
            )

    def test_acceptance_demo_with_mock_llm_five_passes_complete(self):
        """Replicate acceptance.py logic using MockLLM; pipeline must return 5 results."""
        from writing_assistant.style import StyleProfile

        STYLE_SAMPLE = (
            "We build tools that respect the reader. Every sentence earns its place. "
            "However, brevity never excuses vagueness; therefore, each claim is concrete. "
            "Furthermore, we prefer the active voice and plain words."
        )
        # Deliberately wordy draft matching acceptance.py's SAMPLE_DRAFT opening
        SAMPLE_DRAFT = (
            "In the event that you are considering making utilization of our software product, "
            "it is of the utmost importance to take into consideration the fact that there are "
            "numerous features implemented in order to facilitate improvement of written content."
        )
        PASSES = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]

        profile = StyleProfile.learn([STYLE_SAMPLE])
        # MockLLM returns a short, clean response for every pass:
        # - shorter than SAMPLE_DRAFT (word count check passes)
        # - contains no "in the event that" (clarity assertion passes)
        backend = MockLLM(["Short clean rewrite."] * len(PASSES))
        pipeline = Pipeline(passes=PASSES, backend=backend, style_profile=profile)

        results = pipeline.run(SAMPLE_DRAFT)

        assert len(results) == len(PASSES), (
            f"Expected {len(PASSES)} results, got {len(results)}"
        )

    def test_acceptance_demo_mock_llm_final_output_satisfies_acceptance_assertions(self):
        """The MockLLM-backed pipeline must satisfy acceptance.py's own two assertions."""
        from writing_assistant.style import StyleProfile

        SAMPLE_DRAFT = (
            "In the event that you are considering making utilization of our software product, "
            "it is of the utmost importance to take into consideration the fact that there are "
            "numerous features implemented in order to facilitate improvement of written content."
        )
        PASSES = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]

        backend = MockLLM(["Short output."] * len(PASSES))
        results = Pipeline(passes=PASSES, backend=backend).run(SAMPLE_DRAFT)
        final = results[-1].revised

        assert len(final.split()) < len(SAMPLE_DRAFT.split()), (
            "acceptance assertion 1: pipeline must shorten the wordy draft"
        )
        assert "in the event that" not in final.lower(), (
            "acceptance assertion 2: 'in the event that' must not survive to the final output"
        )

    def test_acceptance_demo_subprocess_stderr_is_empty_on_success(self):
        """A clean run must produce no stderr output."""
        result = subprocess.run(
            [sys.executable, str(self._SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stderr.strip() == "", (
            f"acceptance.py wrote to stderr: {result.stderr!r}"
        )
