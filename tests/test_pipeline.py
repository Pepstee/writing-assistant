"""
Adversarial test suite for writing_assistant.pipeline.Pipeline.

Zero real LLM calls: every backend is an instance of MockLLM (or a subclass
that wraps it for call-counting).  The unit under test is Pipeline; MockLLM is
the collaborator, never the SUT.
"""
from __future__ import annotations

import pytest

from mock_llm import MockLLM
from writing_assistant.passes import (
    ADVERSARIAL,
    CLARITY,
    CONCISENESS,
    CONSISTENCY,
    TONE,
)
from writing_assistant.pipeline import Pipeline
from writing_assistant.style import StyleProfile
from writing_assistant.types import Pass, RewriteResult


# ── Test constants ─────────────────────────────────────────────────────────────

DEFAULT_PASSES: list[Pass] = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]
FOUR_NAMED_PASSES: list[Pass] = [CLARITY, TONE, CONCISENESS, CONSISTENCY]
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog.\n"

# Unique sentinel responses — one per pass in DEFAULT_PASSES order.
# Each string is distinct so tests can pinpoint which pass produced which output.
PASS_SENTINELS: dict[str, str] = {
    "clarity":     "Clarity-rewrite sentinel ALPHA.\n",
    "tone":        "Tone-rewrite sentinel BETA.\n",
    "conciseness": "Conciseness-rewrite sentinel GAMMA.\n",
    "consistency": "Consistency-rewrite sentinel DELTA.\n",
    "adversarial": "Adversarial-rewrite sentinel EPSILON.\n",
}


def _sentinels_for(passes: list[Pass]) -> list[str]:
    return [PASS_SENTINELS[p.name] for p in passes]


# ── CountingMockLLM: extends MockLLM to record calls ──────────────────────────

class CountingMockLLM(MockLLM):
    """MockLLM subclass that records how many times generate() was called."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.call_count: int = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return super().generate(prompt)


# ── TestEachPassFiresExactlyOnce ───────────────────────────────────────────────

class TestEachPassFiresExactlyOnce:
    """Each named pass must invoke the LLM exactly once per pipeline run."""

    def test_four_named_passes_each_fire_once(self):
        llm = CountingMockLLM(_sentinels_for(FOUR_NAMED_PASSES))
        Pipeline(passes=FOUR_NAMED_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert llm.call_count == 4, (
            f"Expected 4 generate() calls for 4 passes, got {llm.call_count}"
        )

    def test_five_pass_pipeline_fires_five_times(self):
        llm = CountingMockLLM(_sentinels_for(DEFAULT_PASSES))
        Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert llm.call_count == 5

    def test_single_pass_fires_exactly_once(self):
        llm = CountingMockLLM(["single response\n"])
        Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert llm.call_count == 1

    def test_clarity_produces_nonempty_diff_when_text_changes(self):
        llm = MockLLM(["Completely different clarity output.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_tone_produces_nonempty_diff_when_text_changes(self):
        llm = MockLLM(["Completely different tone output.\n"])
        results = Pipeline(passes=[TONE], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_conciseness_produces_nonempty_diff_when_text_changes(self):
        llm = MockLLM(["Short.\n"])
        results = Pipeline(passes=[CONCISENESS], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_consistency_produces_nonempty_diff_when_text_changes(self):
        llm = MockLLM(["Consistent output.\n"])
        results = Pipeline(passes=[CONSISTENCY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_all_four_named_passes_produce_nonempty_diff_in_one_run(self):
        llm = MockLLM(_sentinels_for(FOUR_NAMED_PASSES))
        results = Pipeline(passes=FOUR_NAMED_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert len(results) == 4
        for p, result in zip(FOUR_NAMED_PASSES, results):
            assert result.diff != "", (
                f"Pass '{p.name}' produced an empty diff, but the text changed"
            )

    def test_result_count_equals_pass_count(self):
        for n in range(1, len(DEFAULT_PASSES) + 1):
            passes = DEFAULT_PASSES[:n]
            llm = MockLLM(_sentinels_for(passes))
            results = Pipeline(passes=passes, backend=llm).run(SAMPLE_TEXT)
            assert len(results) == n, (
                f"Pipeline with {n} passes returned {len(results)} results"
            )

    def test_second_run_fires_again_independently(self):
        """Each run is independent; a second run fires passes again."""
        llm = CountingMockLLM(
            _sentinels_for(FOUR_NAMED_PASSES) + _sentinels_for(FOUR_NAMED_PASSES)
        )
        pipeline = Pipeline(passes=FOUR_NAMED_PASSES, backend=llm)
        pipeline.run(SAMPLE_TEXT)
        pipeline.run(SAMPLE_TEXT)
        assert llm.call_count == 8


# ── TestAdversarialPassIsLast ─────────────────────────────────────────────────

class TestAdversarialPassIsLast:
    """Adversarial pass must occupy the last slot and its result must be returned."""

    def test_adversarial_result_is_last_in_returned_list(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert results[-1].revised == PASS_SENTINELS["adversarial"]

    def test_default_passes_have_adversarial_at_index_four(self):
        assert DEFAULT_PASSES[-1] is ADVERSARIAL

    def test_adversarial_result_present_in_returned_list(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        adversarial_revised = [r.revised for r in results]
        assert PASS_SENTINELS["adversarial"] in adversarial_revised

    def test_adversarial_result_is_only_the_last_element(self):
        """The adversarial sentinel must appear exactly once, at the end."""
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        sentinel = PASS_SENTINELS["adversarial"]
        matches = [i for i, r in enumerate(results) if r.revised == sentinel]
        assert matches == [len(DEFAULT_PASSES) - 1], (
            f"Adversarial sentinel appeared at positions {matches}, "
            f"expected only [{len(DEFAULT_PASSES) - 1}]"
        )

    def test_non_adversarial_passes_precede_adversarial(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        # First 4 results are not the adversarial sentinel
        for i, result in enumerate(results[:-1]):
            assert result.revised != PASS_SENTINELS["adversarial"], (
                f"Adversarial sentinel appeared at position {i}, before the last slot"
            )

    def test_adversarial_pass_result_has_nonempty_diff_when_text_changes(self):
        """Adversarial pass produces a diff when its output differs from what it received."""
        responses = _sentinels_for(DEFAULT_PASSES)
        # All responses are distinct strings — every pass including adversarial changes text
        llm = MockLLM(responses)
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert results[-1].diff != ""


# ── TestMutationResistanceOnPassOrdering ──────────────────────────────────────

class TestMutationResistanceOnPassOrdering:
    """Removing any single pass from DEFAULT_PASSES must remove its RewriteResult."""

    @pytest.mark.parametrize("removed_name", [
        "clarity", "tone", "conciseness", "consistency", "adversarial",
    ])
    def test_removing_pass_reduces_result_count_by_one(self, removed_name: str):
        reduced = [p for p in DEFAULT_PASSES if p.name != removed_name]
        llm = MockLLM(_sentinels_for(reduced))
        results = Pipeline(passes=reduced, backend=llm).run(SAMPLE_TEXT)
        assert len(results) == len(DEFAULT_PASSES) - 1, (
            f"Expected {len(DEFAULT_PASSES) - 1} results after removing '{removed_name}', "
            f"got {len(results)}"
        )

    @pytest.mark.parametrize("removed_name", [
        "clarity", "tone", "conciseness", "consistency", "adversarial",
    ])
    def test_removed_pass_sentinel_absent_from_revised_texts(self, removed_name: str):
        """The sentinel response earmarked for the removed pass must not appear anywhere."""
        reduced = [p for p in DEFAULT_PASSES if p.name != removed_name]
        llm = MockLLM(_sentinels_for(reduced))
        results = Pipeline(passes=reduced, backend=llm).run(SAMPLE_TEXT)
        removed_sentinel = PASS_SENTINELS[removed_name]
        all_revised = [r.revised for r in results]
        assert removed_sentinel not in all_revised, (
            f"Sentinel for removed pass '{removed_name}' appeared in results: {all_revised}"
        )

    @pytest.mark.parametrize("removed_name", [
        "clarity", "tone", "conciseness", "consistency", "adversarial",
    ])
    def test_surviving_passes_sentinels_present_in_order(self, removed_name: str):
        """Every surviving pass must produce its sentinel response, in pass order."""
        reduced = [p for p in DEFAULT_PASSES if p.name != removed_name]
        llm = MockLLM(_sentinels_for(reduced))
        results = Pipeline(passes=reduced, backend=llm).run(SAMPLE_TEXT)
        for i, p in enumerate(reduced):
            assert results[i].revised == PASS_SENTINELS[p.name], (
                f"Surviving pass '{p.name}' at index {i} has wrong revised text"
            )

    def test_removing_only_pass_yields_empty_result_list(self):
        """A pipeline with zero passes returns an empty list."""
        llm = MockLLM([])
        results = Pipeline(passes=[], backend=llm).run(SAMPLE_TEXT)
        assert results == []

    def test_full_pipeline_has_five_results(self):
        """Baseline: DEFAULT_PASSES produces exactly 5 results."""
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert len(results) == 5


# ── TestUnifiedDiffFormat ─────────────────────────────────────────────────────

class TestUnifiedDiffFormat:
    """Diff fields must use standard unified-diff format when text changes."""

    def test_diff_contains_triple_minus_line(self):
        llm = MockLLM(["Completely different output.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert "---" in results[0].diff, "Missing '---' in unified diff output"

    def test_diff_contains_triple_plus_line(self):
        llm = MockLLM(["Completely different output.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert "+++" in results[0].diff, "Missing '+++' in unified diff output"

    def test_diff_triple_minus_and_plus_present_for_all_changing_passes(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        for i, (p, result) in enumerate(zip(DEFAULT_PASSES, results)):
            assert "---" in result.diff, (
                f"Pass '{p.name}' (index {i}) diff missing '---'"
            )
            assert "+++" in result.diff, (
                f"Pass '{p.name}' (index {i}) diff missing '+++'"
            )

    def test_diff_contains_hunk_header(self):
        llm = MockLLM(["Different text entirely.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert "@@" in results[0].diff

    def test_diff_minus_lines_contain_removed_text(self):
        original = "remove this line\n"
        revised = "add this line\n"
        llm = MockLLM([revised])
        results = Pipeline(passes=[CLARITY], backend=llm).run(original)
        minus_lines = [
            line for line in results[0].diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        assert any("remove this line" in line for line in minus_lines)

    def test_diff_plus_lines_contain_added_text(self):
        original = "remove this line\n"
        revised = "add this line\n"
        llm = MockLLM([revised])
        results = Pipeline(passes=[CLARITY], backend=llm).run(original)
        plus_lines = [
            line for line in results[0].diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        assert any("add this line" in line for line in plus_lines)

    def test_diff_empty_when_text_unchanged(self):
        llm = MockLLM([SAMPLE_TEXT])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff == ""

    def test_diff_not_empty_when_text_changed(self):
        llm = MockLLM(["Totally different.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_diff_fromfile_label_is_input(self):
        llm = MockLLM(["Changed text.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert "input" in results[0].diff

    def test_diff_tofile_label_is_revised(self):
        llm = MockLLM(["Changed text.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert "revised" in results[0].diff


# ── TestRewriteResultFields ───────────────────────────────────────────────────

class TestRewriteResultFields:
    """RewriteResult original/revised fields must be correctly populated."""

    def test_first_result_original_is_initial_text(self):
        llm = MockLLM(_sentinels_for(FOUR_NAMED_PASSES))
        results = Pipeline(passes=FOUR_NAMED_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert results[0].original == SAMPLE_TEXT

    def test_each_result_original_is_previous_result_revised(self):
        """Passes chain: result[i].original must equal result[i-1].revised."""
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        for i in range(1, len(results)):
            assert results[i].original == results[i - 1].revised, (
                f"results[{i}].original != results[{i - 1}].revised — "
                "passes are not chained correctly"
            )

    def test_each_result_revised_matches_mock_response(self):
        responses = _sentinels_for(DEFAULT_PASSES)
        llm = MockLLM(responses)
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        for i, (expected, result) in enumerate(zip(responses, results)):
            assert result.revised == expected, (
                f"results[{i}].revised != expected sentinel '{expected}'"
            )

    def test_result_is_rewrite_result_instance(self):
        llm = MockLLM(["output\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert isinstance(results[0], RewriteResult)

    def test_original_field_not_overwritten_with_revised(self):
        llm = MockLLM(["REVISED_SENTINEL\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].original == SAMPLE_TEXT
        assert results[0].revised == "REVISED_SENTINEL\n"
        assert results[0].original != results[0].revised


# ── TestAdversarialPassPromptConstruction ─────────────────────────────────────

class TestAdversarialPassPromptConstruction:
    """Adversarial pass must receive a special prompt with accumulated history."""

    def test_adversarial_prompt_contains_original_text(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        responses = _sentinels_for([CLARITY, ADVERSARIAL])
        llm = RecordingLLM(responses)
        Pipeline(passes=[CLARITY, ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)

        adversarial_prompt = prompts[1]
        assert SAMPLE_TEXT.strip() in adversarial_prompt or "Original text" in adversarial_prompt

    def test_adversarial_prompt_contains_accumulated_rewrites_header(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        responses = _sentinels_for([CLARITY, ADVERSARIAL])
        llm = RecordingLLM(responses)
        Pipeline(passes=[CLARITY, ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)

        adversarial_prompt = prompts[1]
        assert "Accumulated rewrites" in adversarial_prompt

    def test_adversarial_prompt_includes_clarity_pass_name(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        responses = _sentinels_for([CLARITY, ADVERSARIAL])
        llm = RecordingLLM(responses)
        Pipeline(passes=[CLARITY, ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)

        adversarial_prompt = prompts[1]
        assert "clarity" in adversarial_prompt

    def test_non_adversarial_prompt_does_not_contain_accumulated_rewrites(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        llm = RecordingLLM(_sentinels_for([CLARITY, TONE]))
        Pipeline(passes=[CLARITY, TONE], backend=llm).run(SAMPLE_TEXT)

        for prompt in prompts:
            assert "Accumulated rewrites" not in prompt

    def test_adversarial_prompt_includes_previous_revision_text(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        clarity_sentinel = PASS_SENTINELS["clarity"]
        responses = [clarity_sentinel, PASS_SENTINELS["adversarial"]]
        llm = RecordingLLM(responses)
        Pipeline(passes=[CLARITY, ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)

        adversarial_prompt = prompts[1]
        assert clarity_sentinel.strip() in adversarial_prompt


# ── TestStyleProfileIntegration ───────────────────────────────────────────────

class TestStyleProfileIntegration:
    """Consistency pass must incorporate the StyleProfile summary when provided."""

    def test_consistency_prompt_contains_style_profile_summary(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        profile = StyleProfile.learn(["Therefore the system is fast. However it is slow."])
        llm = RecordingLLM(["style output\n"])
        Pipeline(passes=[CONSISTENCY], backend=llm, style_profile=profile).run(SAMPLE_TEXT)

        assert "Style profile" in prompts[0]

    def test_consistency_prompt_without_profile_has_no_style_profile_header(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        llm = RecordingLLM(["output\n"])
        Pipeline(passes=[CONSISTENCY], backend=llm, style_profile=None).run(SAMPLE_TEXT)

        assert "Style profile" not in prompts[0]

    def test_non_consistency_pass_prompt_unaffected_by_style_profile(self):
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        profile = StyleProfile.learn(["Short simple text."])
        llm = RecordingLLM(["clarity output\n"])
        Pipeline(passes=[CLARITY], backend=llm, style_profile=profile).run(SAMPLE_TEXT)

        assert "Style profile" not in prompts[0]


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_empty_input_text_accepted(self):
        llm = MockLLM(["non-empty output\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run("")
        assert results[0].original == ""

    def test_empty_passes_list_returns_empty_list(self):
        llm = MockLLM([])
        results = Pipeline(passes=[], backend=llm).run(SAMPLE_TEXT)
        assert results == []

    def test_adversarial_as_only_pass_returns_one_result(self):
        llm = MockLLM(["adversarial only output\n"])
        results = Pipeline(passes=[ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)
        assert len(results) == 1

    def test_adversarial_as_only_pass_result_has_diff_when_text_changes(self):
        llm = MockLLM(["adversarial only output\n"])
        results = Pipeline(passes=[ADVERSARIAL], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff != ""

    def test_multiline_text_diffs_correctly(self):
        original = "First line.\nSecond line.\nThird line.\n"
        revised = "First line.\nChanged second line.\nThird line.\n"
        llm = MockLLM([revised])
        results = Pipeline(passes=[CLARITY], backend=llm).run(original)
        assert "-Second line." in results[0].diff
        assert "+Changed second line." in results[0].diff

    def test_identical_input_and_output_yields_empty_diff(self):
        llm = MockLLM([SAMPLE_TEXT])
        results = Pipeline(passes=[CLARITY], backend=llm).run(SAMPLE_TEXT)
        assert results[0].diff == ""

    def test_results_list_is_a_list(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        assert isinstance(results, list)

    def test_all_results_are_rewrite_result_instances(self):
        llm = MockLLM(_sentinels_for(DEFAULT_PASSES))
        results = Pipeline(passes=DEFAULT_PASSES, backend=llm).run(SAMPLE_TEXT)
        for result in results:
            assert isinstance(result, RewriteResult)

    def test_single_word_text_processes_without_error(self):
        llm = MockLLM(["word\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run("hello\n")
        assert len(results) == 1

    def test_text_without_trailing_newline_works(self):
        text_no_newline = "No newline at end"
        llm = MockLLM(["Different text no newline"])
        results = Pipeline(passes=[CLARITY], backend=llm).run(text_no_newline)
        assert results[0].original == text_no_newline


# ── TestMockLLMBehavior ───────────────────────────────────────────────────────

class TestMockLLMBehavior:
    """Verify MockLLM properties relied upon by the pipeline tests above."""

    def test_mock_llm_cycles_through_list_in_order(self):
        llm = MockLLM(["first", "second", "third"])
        assert llm.generate("x") == "first"
        assert llm.generate("x") == "second"
        assert llm.generate("x") == "third"

    def test_mock_llm_wraps_around_after_last(self):
        llm = MockLLM(["a", "b"])
        assert llm.generate("x") == "a"
        assert llm.generate("x") == "b"
        assert llm.generate("x") == "a"

    def test_mock_llm_single_response_repeated(self):
        llm = MockLLM(["only"])
        for _ in range(6):
            assert llm.generate("prompt") == "only"

    def test_mock_llm_dict_mode_returns_matching_value(self):
        llm = MockLLM({"exact prompt": "exact response"})
        assert llm.generate("exact prompt") == "exact response"

    def test_mock_llm_dict_mode_returns_default_for_unknown_prompt(self):
        llm = MockLLM({"known": "value"})
        result = llm.generate("unknown prompt")
        assert result == "mock response"

    def test_two_mock_llm_instances_are_independent(self):
        llm1 = MockLLM(["a", "b"])
        llm2 = MockLLM(["x", "y"])
        assert llm1.generate("p") == "a"
        assert llm2.generate("p") == "x"
        assert llm1.generate("p") == "b"
        assert llm2.generate("p") == "y"

    def test_counting_mock_llm_inherits_from_mock_llm(self):
        llm = CountingMockLLM(["r"])
        assert isinstance(llm, MockLLM)

    def test_counting_mock_llm_records_call_count(self):
        llm = CountingMockLLM(["r1", "r2", "r3"])
        llm.generate("a")
        llm.generate("b")
        llm.generate("c")
        assert llm.call_count == 3

    def test_counting_mock_llm_records_prompts(self):
        llm = CountingMockLLM(["r"])
        llm.generate("prompt_one")
        llm.generate("prompt_two")
        assert llm.prompts == ["prompt_one", "prompt_two"]
