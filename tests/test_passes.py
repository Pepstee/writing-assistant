"""
Adversarial test suite for writing_assistant.passes and supporting backends.

Rules:
- No mocking of the unit under test (MockBackend IS the collaborator, not the SUT).
- Every assertion can fail if the implementation is wrong.
- Covers: normal paths, edge cases, error paths, boundaries, mutation resistance.
"""
from __future__ import annotations

import pytest

from writing_assistant.backends import MockBackend
from writing_assistant.passes import (
    AdversarialPass,
    ClarityPass,
    ConcisenessPass,
    ConsistencyPass,
    PassResult,
    TonePass,
    _BasePass,
    _unified_diff,
)


# ── _unified_diff ─────────────────────────────────────────────────────────────

class TestUnifiedDiff:
    def test_identical_strings_produce_empty_diff(self):
        assert _unified_diff("hello", "hello") == ""

    def test_both_empty_produce_empty_diff(self):
        assert _unified_diff("", "") == ""

    def test_changed_line_produces_nonempty_diff(self):
        assert _unified_diff("foo\n", "bar\n") != ""

    def test_diff_marks_removed_line_with_minus(self):
        diff = _unified_diff("removed\n", "added\n")
        assert "-removed" in diff

    def test_diff_marks_added_line_with_plus(self):
        diff = _unified_diff("removed\n", "added\n")
        assert "+added" in diff

    def test_diff_names_fromfile_original(self):
        diff = _unified_diff("a\n", "b\n")
        assert "original" in diff

    def test_diff_names_tofile_rewritten(self):
        diff = _unified_diff("a\n", "b\n")
        assert "rewritten" in diff

    def test_diff_contains_unified_hunk_header(self):
        diff = _unified_diff("x\n", "y\n")
        assert "@@" in diff

    def test_single_character_change_detected(self):
        assert _unified_diff("abc\n", "axc\n") != ""

    def test_multiline_change_shows_both_sides(self):
        original = "line one\nline two\nline three\n"
        rewritten = "line one\nLINE TWO CHANGED\nline three\n"
        diff = _unified_diff(original, rewritten)
        assert "-line two" in diff
        assert "+LINE TWO CHANGED" in diff

    def test_added_trailing_line_detected(self):
        assert _unified_diff("only\n", "only\nextra\n") != ""

    def test_removed_trailing_line_detected(self):
        assert _unified_diff("line one\nline two\n", "line one\n") != ""

    def test_return_type_is_str(self):
        assert isinstance(_unified_diff("a", "b"), str)

    def test_order_matters_not_symmetric(self):
        """diff(a, b) must differ from diff(b, a) — order is load-bearing."""
        a, b = "alpha\n", "beta\n"
        assert _unified_diff(a, b) != _unified_diff(b, a)

    def test_deterministic_for_same_inputs(self):
        a, b = "foo\n", "bar\n"
        assert _unified_diff(a, b) == _unified_diff(a, b)

    def test_only_original_empty_shows_addition(self):
        diff = _unified_diff("", "new content\n")
        assert "+" in diff

    def test_only_rewritten_empty_shows_deletion(self):
        diff = _unified_diff("old content\n", "")
        assert "-" in diff


# ── PassResult dataclass ──────────────────────────────────────────────────────

class TestPassResult:
    def test_success_defaults_to_true(self):
        r = PassResult(original="o", rewritten="r", diff="d")
        assert r.success is True

    def test_notes_defaults_to_empty_string(self):
        r = PassResult(original="o", rewritten="r", diff="d")
        assert r.notes == ""

    def test_all_fields_stored_verbatim(self):
        r = PassResult(original="orig", rewritten="new", diff="@@", notes="note", success=False)
        assert r.original == "orig"
        assert r.rewritten == "new"
        assert r.diff == "@@"
        assert r.notes == "note"
        assert r.success is False

    def test_original_field_distinct_from_rewritten(self):
        r = PassResult(original="ORIGINAL", rewritten="REWRITTEN", diff="")
        assert r.original != r.rewritten

    def test_success_can_be_set_false(self):
        r = PassResult(original="o", rewritten="r", diff="d", success=False)
        assert r.success is False

    def test_diff_field_stored_exactly(self):
        r = PassResult(original="o", rewritten="r", diff="exact diff content")
        assert r.diff == "exact diff content"


# ── MockBackend ───────────────────────────────────────────────────────────────

class TestMockBackend:
    def test_returns_configured_response(self):
        backend = MockBackend(["expected"])
        assert backend.complete("any prompt") == "expected"

    def test_default_response_is_mock_response(self):
        backend = MockBackend()
        assert backend.complete("anything") == "mock response"

    def test_cycles_through_multiple_responses_in_order(self):
        backend = MockBackend(["first", "second", "third"])
        assert backend.complete("p") == "first"
        assert backend.complete("p") == "second"
        assert backend.complete("p") == "third"

    def test_wraps_around_after_last_response(self):
        backend = MockBackend(["a", "b"])
        assert backend.complete("p") == "a"
        assert backend.complete("p") == "b"
        assert backend.complete("p") == "a"  # cycle restarts

    def test_prompt_content_does_not_affect_response(self):
        backend = MockBackend(["fixed"])
        assert backend.complete("prompt one") == "fixed"
        assert backend.complete("completely different prompt") == "fixed"

    def test_single_response_repeated_indefinitely(self):
        backend = MockBackend(["only"])
        for _ in range(10):
            assert backend.complete("p") == "only"

    def test_returns_response_not_prompt(self):
        backend = MockBackend(["the response"])
        prompt = "the prompt — totally different"
        result = backend.complete(prompt)
        assert result == "the response"
        assert result != prompt

    def test_two_instances_have_independent_state(self):
        b1 = MockBackend(["a", "b"])
        b2 = MockBackend(["x", "y"])
        assert b1.complete("p") == "a"
        assert b2.complete("p") == "x"
        assert b1.complete("p") == "b"
        assert b2.complete("p") == "y"


# ── _BasePass.run() ───────────────────────────────────────────────────────────

class _CapturingBackend:
    """Records the last prompt given; returns a fixed response."""
    def __init__(self, response: str = "ok") -> None:
        self._response = response
        self.last_prompt: str = ""
        self.call_count: int = 0

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        return self._response


class TestBasePassRun:
    def test_run_returns_pass_result_instance(self):
        result = ClarityPass().run("text", MockBackend(["new"]))
        assert isinstance(result, PassResult)

    def test_original_preserved_exactly(self):
        text = "the original immutable input"
        result = ClarityPass().run(text, MockBackend(["anything"]))
        assert result.original == text

    def test_rewritten_comes_from_backend(self):
        result = ClarityPass().run("text", MockBackend(["backend output"]))
        assert result.rewritten == "backend output"

    def test_prompt_contains_instructions(self):
        cap = _CapturingBackend()
        p = ClarityPass()
        p.run("some text", cap)
        assert p.INSTRUCTIONS in cap.last_prompt

    def test_prompt_contains_input_text(self):
        cap = _CapturingBackend()
        unique = "very specific unique text 98765xyz"
        ClarityPass().run(unique, cap)
        assert unique in cap.last_prompt

    def test_backend_called_exactly_once_per_run(self):
        cap = _CapturingBackend()
        ClarityPass().run("text", cap)
        assert cap.call_count == 1

    def test_diff_nonempty_when_rewritten_differs(self):
        result = ClarityPass().run("original input\n", MockBackend(["completely different output\n"]))
        assert result.diff != ""

    def test_diff_empty_when_rewritten_matches_original(self):
        text = "unchanged text"
        result = ClarityPass().run(text, MockBackend([text]))
        assert result.diff == ""

    def test_diff_equals_computed_unified_diff(self):
        """The stored diff must equal _unified_diff(original, rewritten) — not cached/stale."""
        original = "old content\n"
        rewritten = "new content\n"
        result = ClarityPass().run(original, MockBackend([rewritten]))
        assert result.diff == _unified_diff(original, rewritten)

    def test_success_is_true_by_default(self):
        result = ClarityPass().run("text", MockBackend(["anything"]))
        assert result.success is True

    def test_empty_input_text_accepted(self):
        result = ClarityPass().run("", MockBackend(["something"]))
        assert result.original == ""

    def test_multiline_original_preserved_intact(self):
        text = "line one\nline two\nline three"
        result = ClarityPass().run(text, MockBackend(["short"]))
        assert result.original == text

    def test_each_run_consumes_exactly_one_backend_response(self):
        """Second run gets the second response, not the first again."""
        backend = MockBackend(["first", "second"])
        ClarityPass().run("text", backend)
        result = ClarityPass().run("text", backend)
        assert result.rewritten == "second"

    def test_original_not_modified_by_pass(self):
        """Pass must not mutate the string passed as input."""
        original = "unchanged original"
        backend = MockBackend(["modified version"])
        result = ClarityPass().run(original, backend)
        assert result.original == "unchanged original"


# ── Per-pass instruction distinctiveness ─────────────────────────────────────

ALL_PASSES = [ClarityPass, TonePass, ConcisenessPass, ConsistencyPass, AdversarialPass]


class TestPassInstructions:
    def test_each_pass_has_nonempty_instructions(self):
        for cls in ALL_PASSES:
            assert cls.INSTRUCTIONS.strip() != "", f"{cls.__name__}.INSTRUCTIONS is empty"

    def test_all_passes_have_distinct_instructions(self):
        instructions = [cls.INSTRUCTIONS for cls in ALL_PASSES]
        assert len(instructions) == len(set(instructions)), (
            "Two or more passes share identical INSTRUCTIONS"
        )

    def test_clarity_pass_mentions_clarity_concept(self):
        assert "clarity" in ClarityPass.INSTRUCTIONS.lower()

    def test_tone_pass_mentions_tone(self):
        assert "tone" in TonePass.INSTRUCTIONS.lower()

    def test_conciseness_pass_mentions_concise(self):
        instr = ConcisenessPass.INSTRUCTIONS.lower()
        assert "concise" in instr or "conciseness" in instr

    def test_consistency_pass_mentions_consistent(self):
        instr = ConsistencyPass.INSTRUCTIONS.lower()
        assert "consistent" in instr or "consistency" in instr

    def test_adversarial_pass_mentions_adversarial_concept(self):
        instr = AdversarialPass.INSTRUCTIONS.lower()
        assert any(word in instr for word in ("adversarial", "weakness", "weaknesses", "flaw"))

    def test_adversarial_pass_instructs_improvement(self):
        instr = AdversarialPass.INSTRUCTIONS.lower()
        assert "improve" in instr or "improved" in instr

    def test_all_passes_are_subclasses_of_base_pass(self):
        for cls in ALL_PASSES:
            assert issubclass(cls, _BasePass), f"{cls.__name__} is not a subclass of _BasePass"


# ── AdversarialPass: planted-flaw detection ───────────────────────────────────

class TestAdversarialPassFlawDetection:
    """Adversarial pass must surface diffs when given text with planted weaknesses."""

    def test_planted_redundancy_yields_nonempty_diff(self):
        flawed = (
            "The thing is very very good and it is also quite quite nice "
            "and additionally moreover furthermore it has good qualities.\n"
        )
        improved = "The product is excellent and pleasant, with many fine qualities.\n"
        result = AdversarialPass().run(flawed, MockBackend([improved]))
        assert result.diff != ""

    def test_planted_jargon_yields_nonempty_diff(self):
        jargon_heavy = (
            "Leverage synergies to optimise the paradigm shift via cross-functional "
            "stakeholder engagement and holistic value-add deliverables.\n"
        )
        plain = "Work together across teams to create real value.\n"
        result = AdversarialPass().run(jargon_heavy, MockBackend([plain]))
        assert result.diff != ""

    def test_planted_logical_gap_yields_nonempty_diff(self):
        gappy = "We should switch to product X. Therefore profits will double.\n"
        filled = (
            "We should switch to product X because its cost is half of Y. "
            "This reduction in expenses should improve margins.\n"
        )
        result = AdversarialPass().run(gappy, MockBackend([filled]))
        assert result.diff != ""

    def test_original_is_the_flawed_text(self):
        flawed = "Unclear, redundant, and jargon-laden prose everywhere.\n"
        improved = "Clear, concise prose.\n"
        result = AdversarialPass().run(flawed, MockBackend([improved]))
        assert result.original == flawed

    def test_rewritten_is_the_improved_text(self):
        flawed = "Very very bad unclear text.\n"
        improved = "Clear text.\n"
        result = AdversarialPass().run(flawed, MockBackend([improved]))
        assert result.rewritten == improved

    def test_diff_content_reflects_actual_change(self):
        original = "bad sentence here\n"
        rewritten = "good sentence here\n"
        result = AdversarialPass().run(original, MockBackend([rewritten]))
        assert "-bad sentence here" in result.diff
        assert "+good sentence here" in result.diff

    def test_no_phantom_diff_when_text_is_already_good(self):
        """If backend echoes the same text, diff must be empty — no hallucinated fixes."""
        good = "A clear, concise, professional sentence with no flaws."
        result = AdversarialPass().run(good, MockBackend([good]))
        assert result.diff == ""

    def test_adversarial_pass_still_stores_original_not_rewritten(self):
        original = "ORIGINAL TEXT"
        rewritten = "REWRITTEN TEXT"
        result = AdversarialPass().run(original, MockBackend([rewritten]))
        assert result.original == "ORIGINAL TEXT"
        assert result.rewritten == "REWRITTEN TEXT"


# ── Per-pass: diff non-empty on real change, empty on identity ────────────────

@pytest.mark.parametrize("pass_cls", ALL_PASSES)
class TestPerPassDiffContract:
    def test_diff_nonempty_on_real_change(self, pass_cls):
        original = "The utilisation of unnecessarily verbose language everywhere.\n"
        rewritten = "Use clear, simple language.\n"
        result = pass_cls().run(original, MockBackend([rewritten]))
        assert result.diff != "", (
            f"{pass_cls.__name__}: diff must be non-empty when rewritten differs from original"
        )

    def test_diff_empty_when_text_unchanged(self, pass_cls):
        text = "Stable text that will not change at all.\n"
        result = pass_cls().run(text, MockBackend([text]))
        assert result.diff == "", (
            f"{pass_cls.__name__}: diff must be empty when rewritten matches original"
        )

    def test_original_never_overwritten_with_rewritten(self, pass_cls):
        original = "immutable original text"
        result = pass_cls().run(original, MockBackend(["completely different rewritten version"]))
        assert result.original == original, (
            f"{pass_cls.__name__}: original field must not contain the rewritten text"
        )

    def test_rewritten_comes_from_backend_not_input(self, pass_cls):
        backend_response = "BACKEND_OUTPUT_SENTINEL_12345"
        result = pass_cls().run("input text", MockBackend([backend_response]))
        assert result.rewritten == backend_response, (
            f"{pass_cls.__name__}: rewritten must be exactly the backend response"
        )


# ── Mutation-resistant: tight invariants that break if implementation drifts ──

class TestMutationResistant:
    def test_pass_result_diff_not_always_empty(self):
        """Guard against a mutation that wipes diff computation."""
        result = ClarityPass().run("old line\n", MockBackend(["new line\n"]))
        assert result.diff != ""

    def test_pass_result_diff_not_always_nonempty(self):
        """Guard against a mutation that returns a dummy non-empty diff for unchanged text."""
        text = "same text"
        result = ClarityPass().run(text, MockBackend([text]))
        assert result.diff == ""

    def test_unified_diff_is_not_just_concatenation(self):
        """Diff must not simply concatenate original + rewritten."""
        a, b = "foo\n", "bar\n"
        diff = _unified_diff(a, b)
        assert diff != a + b

    def test_pass_prompt_is_not_just_instructions(self):
        """Prompt must include the text, not only INSTRUCTIONS."""
        cap = _CapturingBackend()
        unique = "UNIQUE_SENTINEL_TEXT_99"
        ClarityPass().run(unique, cap)
        assert unique in cap.last_prompt

    def test_pass_prompt_is_not_just_input_text(self):
        """Prompt must include INSTRUCTIONS, not only the text."""
        cap = _CapturingBackend()
        p = ClarityPass()
        p.run("some text", cap)
        assert p.INSTRUCTIONS in cap.last_prompt

    def test_mock_backend_cycle_is_not_random(self):
        """Responses cycle deterministically — second call is not first again."""
        backend = MockBackend(["response_A", "response_B"])
        r1 = backend.complete("p")
        r2 = backend.complete("p")
        assert r1 != r2

    def test_diff_direction_correct_original_minus_rewritten_plus(self):
        """Minus sign belongs to original; plus to rewritten — not swapped."""
        diff = _unified_diff("original_word\n", "rewritten_word\n")
        # Lines starting with '-' must contain original content
        minus_lines = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus_lines = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert any("original_word" in l for l in minus_lines)
        assert any("rewritten_word" in l for l in plus_lines)

    def test_pass_result_fields_not_swapped(self):
        """Original and rewritten must not be stored in each other's fields."""
        original = "THE_ORIGINAL_9001"
        rewritten = "THE_REWRITTEN_8999"
        result = TonePass().run(original, MockBackend([rewritten]))
        assert result.original == original
        assert result.rewritten == rewritten
        assert result.original != result.rewritten

    def test_all_pass_classes_instantiate_without_arguments(self):
        for cls in ALL_PASSES:
            instance = cls()
            assert instance is not None

    def test_all_pass_classes_have_run_method(self):
        for cls in ALL_PASSES:
            assert callable(getattr(cls(), "run", None)), f"{cls.__name__} missing callable run()"

    def test_base_pass_instructions_is_empty_string(self):
        """_BasePass.INSTRUCTIONS is deliberately blank — subclasses must override."""
        assert _BasePass.INSTRUCTIONS == ""

    def test_subclass_instructions_override_base(self):
        for cls in ALL_PASSES:
            assert cls.INSTRUCTIONS != _BasePass.INSTRUCTIONS, (
                f"{cls.__name__} did not override INSTRUCTIONS"
            )
