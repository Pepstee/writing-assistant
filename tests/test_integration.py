"""
Integration tests: full pipeline + style-profile wired together.

These tests verify the complete end-to-end flow:
  StyleProfile.learn() → Pipeline (all five passes) → per-pass diffs produced
  → adversarial pass fires last → final result returned

All LLM calls use MockLLM; no real network calls are made.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mock_llm import MockLLM
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.style import StyleProfile
from writing_assistant.types import Pass, RewriteResult

REPO_ROOT = Path(__file__).parent.parent

# ── Shared test data ───────────────────────────────────────────────────────────

SAMPLE_STYLE_TEXT = (
    "The report was analyzed by Alice. Furthermore, errors were detected. "
    "However, the system recovered quickly. Therefore, the project succeeded. "
    "The data were processed and results were obtained."
)

INPUT_DRAFT = (
    "In the event that you are considering making utilization of our software product, "
    "it is of the utmost importance to take into consideration the fact that there are "
    "numerous features which have been implemented in order to facilitate improvement.\n"
)

ALL_PASSES: list[Pass] = [CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL]

SENTINEL_RESPONSES = [
    "Clarity pass output: clear and plain language version.\n",
    "Tone pass output: professional and respectful tone.\n",
    "Conciseness pass output: concise and to the point.\n",
    "Consistency pass output: consistent terminology and voice.\n",
    "Adversarial pass output: self-reviewed and hardened final text.\n",
]


class RecordingMockLLM(MockLLM):
    """MockLLM that records every prompt it receives, in call order."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        response = super().generate(prompt)
        self.calls.append(prompt)
        return response


# ── Full end-to-end flow ───────────────────────────────────────────────────────

class TestFullEndToEndFlow:
    """
    Wire StyleProfile.learn() → Pipeline(all 5 passes + profile + MockLLM) → run().

    Exercises the complete integration path the acceptance criteria require.
    """

    def _full_run(
        self,
    ) -> tuple[StyleProfile, list[RewriteResult], RecordingMockLLM]:
        profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        pipeline = Pipeline(passes=ALL_PASSES, backend=llm, style_profile=profile)
        results = pipeline.run(INPUT_DRAFT)
        return profile, results, llm

    def test_returns_one_result_per_pass(self) -> None:
        _, results, _ = self._full_run()
        assert len(results) == len(ALL_PASSES)

    def test_all_results_are_rewrite_result_instances(self) -> None:
        _, results, _ = self._full_run()
        for i, result in enumerate(results):
            assert isinstance(result, RewriteResult), (
                f"results[{i}] is {type(result)}, expected RewriteResult"
            )

    def test_first_result_original_is_the_input_draft(self) -> None:
        _, results, _ = self._full_run()
        assert results[0].original == INPUT_DRAFT

    def test_passes_chain_each_original_is_previous_revised(self) -> None:
        _, results, _ = self._full_run()
        for i in range(1, len(results)):
            assert results[i].original == results[i - 1].revised, (
                f"Chaining broken at index {i}: "
                f"results[{i}].original != results[{i - 1}].revised"
            )

    def test_each_pass_produces_its_sentinel_response(self) -> None:
        _, results, _ = self._full_run()
        for i, (expected, result) in enumerate(zip(SENTINEL_RESPONSES, results)):
            assert result.revised == expected, (
                f"Pass {i} revised mismatch.\n"
                f"  expected: {expected!r}\n"
                f"  got:      {result.revised!r}"
            )

    def test_final_result_is_adversarial_output(self) -> None:
        _, results, _ = self._full_run()
        assert results[-1].revised == SENTINEL_RESPONSES[-1]

    def test_every_pass_produces_a_nonempty_diff(self) -> None:
        _, results, _ = self._full_run()
        for i, result in enumerate(results):
            assert result.diff != "", (
                f"Pass {i} produced an empty diff — "
                "each sentinel is unique so every pass must change the text"
            )

    def test_llm_called_exactly_once_per_pass(self) -> None:
        _, _, llm = self._full_run()
        assert len(llm.calls) == len(ALL_PASSES)

    def test_style_profile_learned_from_sample_has_nonzero_avg_sentence_length(self) -> None:
        profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        assert profile.avg_sentence_length > 0.0, (
            "StyleProfile.learn() returned zero avg_sentence_length for a non-empty sample"
        )

    def test_style_profile_injected_into_consistency_pass_prompt(self) -> None:
        profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm, style_profile=profile).run(INPUT_DRAFT)
        consistency_prompt = llm.calls[3]  # consistency is the 4th pass (index 3)
        assert "Style profile:" in consistency_prompt, (
            "Consistency pass prompt missing 'Style profile:' header"
        )
        assert profile.summary() in consistency_prompt, (
            "Consistency pass prompt missing the full profile summary text"
        )

    def test_style_profile_not_injected_into_non_consistency_passes(self) -> None:
        profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm, style_profile=profile).run(INPUT_DRAFT)
        for i, (pass_obj, prompt) in enumerate(zip(ALL_PASSES, llm.calls)):
            if pass_obj.name != "consistency":
                assert "Style profile:" not in prompt, (
                    f"'Style profile:' appeared in non-consistency pass '{pass_obj.name}' "
                    f"(index {i})"
                )

    def test_no_real_network_calls_sentinel_responses_match(self) -> None:
        """If MockLLM was bypassed for real I/O, responses would not be sentinels."""
        _, results, _ = self._full_run()
        for i, sentinel in enumerate(SENTINEL_RESPONSES):
            assert results[i].revised == sentinel, (
                f"Pass {i} response is not the MockLLM sentinel — "
                "a real LLM call may have been made"
            )


# ── Per-pass diff verification ─────────────────────────────────────────────────

class TestPerPassDiffsProduced:
    """Every pass must produce a well-formed unified diff when the text changes."""

    def _results(self) -> list[RewriteResult]:
        llm = MockLLM(SENTINEL_RESPONSES)
        return Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)

    def test_each_diff_is_a_string(self) -> None:
        for i, result in enumerate(self._results()):
            assert isinstance(result.diff, str), (
                f"results[{i}].diff is {type(result.diff)}, expected str"
            )

    def test_each_diff_contains_unified_diff_triple_minus(self) -> None:
        for i, result in enumerate(self._results()):
            assert "---" in result.diff, (
                f"Pass {i} diff missing '---' unified-diff header"
            )

    def test_each_diff_contains_unified_diff_triple_plus(self) -> None:
        for i, result in enumerate(self._results()):
            assert "+++" in result.diff, (
                f"Pass {i} diff missing '+++' unified-diff header"
            )

    def test_each_diff_contains_hunk_header(self) -> None:
        for i, result in enumerate(self._results()):
            assert "@@" in result.diff, (
                f"Pass {i} diff missing '@@' hunk header"
            )

    def test_each_diff_has_removed_lines_from_original(self) -> None:
        for i, result in enumerate(self._results()):
            minus_lines = [
                ln for ln in result.diff.splitlines()
                if ln.startswith("-") and not ln.startswith("---")
            ]
            assert minus_lines, (
                f"Pass {i} diff has no '-' lines — original text was not shown as removed"
            )

    def test_each_diff_has_added_lines_for_revised(self) -> None:
        for i, result in enumerate(self._results()):
            plus_lines = [
                ln for ln in result.diff.splitlines()
                if ln.startswith("+") and not ln.startswith("+++")
            ]
            assert plus_lines, (
                f"Pass {i} diff has no '+' lines — revised text was not shown as added"
            )

    def test_adversarial_diff_shows_change_from_consistency_output(self) -> None:
        results = self._results()
        # Adversarial receives SENTINEL_RESPONSES[3] (consistency) and outputs [4]
        assert results[-1].original == SENTINEL_RESPONSES[3]
        assert results[-1].revised == SENTINEL_RESPONSES[4]
        assert results[-1].diff != ""

    def test_diff_empty_when_pass_does_not_change_text(self) -> None:
        unchanged_response = INPUT_DRAFT
        llm = MockLLM([unchanged_response])
        results = Pipeline(passes=[CLARITY], backend=llm).run(INPUT_DRAFT)
        assert results[0].diff == "", (
            "Diff should be empty when the pass returns the exact same text"
        )

    def test_diff_labels_are_input_and_revised(self) -> None:
        llm = MockLLM(["Different output.\n"])
        results = Pipeline(passes=[CLARITY], backend=llm).run("Original text.\n")
        assert "--- input" in results[0].diff, "Diff fromfile label should be 'input'"
        assert "+++ revised" in results[0].diff, "Diff tofile label should be 'revised'"


# ── Adversarial pass fires last ────────────────────────────────────────────────

class TestAdversarialPassFires:
    """Adversarial pass must run last and receive the full accumulated history."""

    def test_adversarial_result_is_the_last_in_the_list(self) -> None:
        llm = MockLLM(SENTINEL_RESPONSES)
        results = Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        assert results[-1].revised == SENTINEL_RESPONSES[4]

    def test_adversarial_fires_exactly_once(self) -> None:
        llm = MockLLM(SENTINEL_RESPONSES)
        results = Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_sentinel = SENTINEL_RESPONSES[4]
        occurrences = [i for i, r in enumerate(results) if r.revised == adversarial_sentinel]
        assert occurrences == [4], (
            f"Adversarial sentinel appeared at indices {occurrences}, expected only [4]"
        )

    def test_adversarial_metadata_flag_is_true(self) -> None:
        assert ADVERSARIAL.metadata.get("adversarial") is True, (
            "ADVERSARIAL pass metadata['adversarial'] must be True"
        )

    def test_adversarial_prompt_contains_original_text(self) -> None:
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_prompt = llm.calls[4]
        assert INPUT_DRAFT.strip() in adversarial_prompt, (
            "Adversarial prompt must embed the original input text"
        )

    def test_adversarial_prompt_contains_accumulated_rewrites_header(self) -> None:
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_prompt = llm.calls[4]
        assert "Accumulated rewrites" in adversarial_prompt

    def test_adversarial_prompt_names_all_prior_passes(self) -> None:
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_prompt = llm.calls[4]
        for pass_name in ("clarity", "tone", "conciseness", "consistency"):
            assert pass_name in adversarial_prompt, (
                f"Adversarial prompt missing reference to prior pass '{pass_name}'"
            )

    def test_adversarial_prompt_includes_each_prior_revised_text(self) -> None:
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_prompt = llm.calls[4]
        for i, sentinel in enumerate(SENTINEL_RESPONSES[:4]):
            assert sentinel.strip() in adversarial_prompt, (
                f"Adversarial prompt missing text produced by pass {i}"
            )

    def test_adversarial_prompt_differs_from_all_non_adversarial_prompts(self) -> None:
        llm = RecordingMockLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm).run(INPUT_DRAFT)
        adversarial_prompt = llm.calls[4]
        for i, prompt in enumerate(llm.calls[:4]):
            assert prompt != adversarial_prompt, (
                f"Adversarial prompt is identical to pass {i} prompt — "
                "the special adversarial path must produce a distinct prompt"
            )

    def test_adversarial_only_pipeline_still_fires(self) -> None:
        llm = MockLLM(["adversarial only output.\n"])
        results = Pipeline(passes=[ADVERSARIAL], backend=llm).run(INPUT_DRAFT)
        assert len(results) == 1
        assert results[0].revised == "adversarial only output.\n"

    def test_adversarial_only_prompt_has_empty_accumulated_history(self) -> None:
        """With no prior passes the history section must be empty (not absent)."""
        prompts: list[str] = []

        class RecordingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        RecordingLLM(["out\n"]).generate  # warm up (unused)
        llm = RecordingLLM(["out\n"])
        Pipeline(passes=[ADVERSARIAL], backend=llm).run("Text.\n")
        assert "Accumulated rewrites" in prompts[0]


# ── CLI entry-point exit-code tests ───────────────────────────────────────────

def _run_cli(*args: str, stdin: str = INPUT_DRAFT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "writing_assistant", *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


class TestCLIExitCode:
    """CLI (python -m writing_assistant) must exit 0 on valid input with --backend rules."""

    def test_cli_exits_zero_with_rules_backend_default_passes(self) -> None:
        result = _run_cli("--backend", "rules")
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_cli_exits_zero_with_single_explicit_pass(self) -> None:
        result = _run_cli("--backend", "rules", "--passes", "clarity")
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_cli_exits_zero_with_all_five_passes_explicit(self) -> None:
        result = _run_cli(
            "--backend", "rules",
            "--passes",
            "clarity,tone,conciseness,consistency,adversarial",
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_cli_exits_zero_with_adversarial_only(self) -> None:
        result = _run_cli("--backend", "rules", "--passes", "adversarial")
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_cli_exits_nonzero_on_empty_stdin(self) -> None:
        result = _run_cli("--backend", "rules", stdin="")
        assert result.returncode != 0, (
            "CLI should exit non-zero when input text is empty"
        )

    def test_cli_exits_nonzero_on_whitespace_only_stdin(self) -> None:
        result = _run_cli("--backend", "rules", stdin="   \n\t  ")
        assert result.returncode != 0, (
            "CLI should exit non-zero when input text is whitespace-only"
        )

    def test_cli_exits_nonzero_on_unknown_pass_name(self) -> None:
        result = _run_cli("--backend", "rules", "--passes", "nosuchpass")
        assert result.returncode != 0, (
            "CLI should exit non-zero when an unknown pass name is given"
        )

    def test_cli_exits_nonzero_on_mixed_valid_and_unknown_passes(self) -> None:
        result = _run_cli("--backend", "rules", "--passes", "clarity,nosuchpass")
        assert result.returncode != 0

    def test_cli_exits_nonzero_when_pass_list_is_empty(self) -> None:
        result = _run_cli("--backend", "rules", "--passes", ", ,")
        assert result.returncode != 0
        assert "must contain at least one pass name" in result.stderr

    def test_cli_stdout_contains_final_draft_header(self) -> None:
        result = _run_cli("--backend", "rules")
        assert result.returncode == 0
        assert "Final draft:" in result.stdout, (
            "CLI output should contain 'Final draft:' section"
        )

    def test_cli_stdout_contains_pass_section_for_each_default_pass(self) -> None:
        result = _run_cli("--backend", "rules")
        assert result.returncode == 0
        for pass_name in ("CLARITY", "TONE", "CONCISENESS", "CONSISTENCY", "ADVERSARIAL"):
            assert f"Pass: {pass_name}" in result.stdout, (
                f"CLI output missing 'Pass: {pass_name}' section header"
            )

    def test_cli_stdout_is_nonempty_on_success(self) -> None:
        result = _run_cli("--backend", "rules")
        assert result.returncode == 0
        assert result.stdout.strip() != "", "CLI produced no output on success"

    def test_cli_stdin_with_leading_whitespace_and_content_exits_zero(self) -> None:
        result = _run_cli("--backend", "rules", stdin="\n\n   Some actual content here.\n\n")
        assert result.returncode == 0, (
            f"CLI should succeed when stdin has actual content after stripping.\n"
            f"stderr: {result.stderr}"
        )

    def test_default_console_output_is_byte_exact(self) -> None:
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity,tone",
            stdin="A very really useful draft that is basically clear.\n",
        )
        expected = (
            "\n============================================================\n"
            "Pass: CLARITY\n"
            "============================================================\n"
            "(no changes)\n"
            "\n============================================================\n"
            "Pass: TONE\n"
            "============================================================\n"
            "(no changes)\n"
            "\n============================================================\n"
            "Final draft:\n"
            "============================================================\n"
            "A very really useful draft that is basically clear.\n"
        )
        assert result.returncode == 0
        assert result.stdout == expected
        assert result.stderr == ""

    def test_markdown_exports_final_text_and_ordered_pass_diffs(self) -> None:
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity,tone,consistency",
            "--format",
            "markdown",
            stdin="In the event that we can't utilize the web site.\n",
        )
        assert result.returncode == 0
        assert "## Final Text" in result.stdout
        assert "If we cannot use the website." in result.stdout
        assert (
            result.stdout.index("### clarity")
            < result.stdout.index("### tone")
            < result.stdout.index("### consistency")
        )
        assert "```diff" in result.stdout
        assert "--- input" in result.stdout
        assert "+++ revised" in result.stdout

    def test_markdown_marks_a_pass_that_makes_no_change(self) -> None:
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity",
            "--format",
            "markdown",
            stdin="Already clear.\n",
        )
        assert result.returncode == 0
        assert "### clarity\n\n```diff\n(no changes)\n```" in result.stdout

    def test_plain_format_emits_only_the_final_text(self) -> None:
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity",
            "--format",
            "plain",
            stdin="In the event that this works.\n",
        )
        assert result.returncode == 0
        assert result.stdout == "If this works.\n"
        assert result.stderr == ""

    def test_output_file_receives_complete_markdown_and_stdout_stays_empty(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "rewrite.md"
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity",
            "--format",
            "markdown",
            "--output",
            str(output),
            stdin="In the event that this works.\n",
        )
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""
        exported = output.read_text(encoding="utf-8")
        assert exported.startswith("## Final Text\n")
        assert "If this works." in exported
        assert "## Pass Diffs" in exported

    def test_output_write_error_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_cli(
            "--backend",
            "rules",
            "--passes",
            "clarity",
            "--output",
            str(tmp_path),
            stdin="Already clear.\n",
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "error writing output" in result.stderr


# ── Acceptance script exit-0 ───────────────────────────────────────────────────

class TestAcceptanceScriptExitCode:
    """acceptance.py must run to completion and exit 0 without network calls."""

    _ACCEPTANCE_PATH = REPO_ROOT / "acceptance.py"

    def test_acceptance_script_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"acceptance.py exited {result.returncode}.\nstderr: {result.stderr}"
        )

    def test_acceptance_script_never_invokes_an_installed_claude_cli(
        self, tmp_path: Path
    ) -> None:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        marker = tmp_path / "claude-was-invoked"
        fake_claude = fake_bin / "claude"
        fake_claude.write_text(
            "#!/bin/sh\nprintf invoked > \"$FAKE_CLAUDE_MARKER\"\nexit 99\n",
            encoding="utf-8",
        )
        fake_claude.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
        environment["FAKE_CLAUDE_MARKER"] = str(marker)

        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
            env=environment,
        )

        assert result.returncode == 0
        assert "rule-based (offline)" in result.stdout
        assert "Claude CLI" not in result.stdout
        assert not marker.exists()

    def test_acceptance_script_produces_final_rewrite_header(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Final rewrite:" in result.stdout, (
            "acceptance.py should print a 'Final rewrite:' section"
        )

    def test_acceptance_script_shows_all_five_pass_names(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        stdout_lower = result.stdout.lower()
        for pass_name in ("clarity", "tone", "conciseness", "consistency", "adversarial"):
            assert pass_name in stdout_lower, (
                f"acceptance.py output missing pass name '{pass_name}'"
            )

    def test_acceptance_script_shows_adversarial_note(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "adversarial" in result.stdout.lower(), (
            "acceptance.py output should reference the adversarial pass"
        )

    def test_acceptance_script_produces_nonempty_output(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self._ACCEPTANCE_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


# ── Style profile → pipeline integration (combined) ───────────────────────────

class TestStyleProfilePipelineIntegration:
    """
    Tests that specifically exercise the style-profile-to-pipeline wiring:
    learn() output must influence the pipeline prompt, not be silently ignored.
    """

    def test_profile_with_transition_words_appears_in_consistency_prompt(self) -> None:
        sample = "However, the test passed. Therefore, we are confident. Moreover, it works."
        profile = StyleProfile.learn([sample])
        assert "however" in profile.preferred_transition_words

        prompts: list[str] = []

        class CapturingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        llm = CapturingLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm, style_profile=profile).run(INPUT_DRAFT)
        consistency_prompt = prompts[3]
        assert "however" in consistency_prompt, (
            "Transition word 'however' from the style profile must appear in the consistency prompt"
        )
        assert "therefore" in consistency_prompt

    def test_profile_avg_sentence_length_appears_in_consistency_prompt(self) -> None:
        profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        prompts: list[str] = []

        class CapturingLLM(MockLLM):
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return super().generate(prompt)

        llm = CapturingLLM(SENTINEL_RESPONSES)
        Pipeline(passes=ALL_PASSES, backend=llm, style_profile=profile).run(INPUT_DRAFT)
        consistency_prompt = prompts[3]
        formatted_length = f"{profile.avg_sentence_length:.1f}"
        assert formatted_length in consistency_prompt, (
            f"avg_sentence_length={formatted_length} not found in consistency prompt"
        )

    def test_different_profiles_produce_different_consistency_prompts(self) -> None:
        profile_a = StyleProfile.learn(["Short text. Short text."])
        profile_b = StyleProfile.learn([
            "The verbose and lengthy text was thoroughly analyzed by the senior researcher. "
            "Furthermore, the comprehensive study was conducted over an extended period."
        ])
        assert profile_a.avg_sentence_length != profile_b.avg_sentence_length

        prompts_a: list[str] = []
        prompts_b: list[str] = []

        class CapturingLLM(MockLLM):
            def __init__(self, store: list[str], responses: list[str]) -> None:
                super().__init__(responses)
                self._store = store

            def generate(self, prompt: str) -> str:
                self._store.append(prompt)
                return super().generate(prompt)

        llm_a = CapturingLLM(prompts_a, SENTINEL_RESPONSES[:])
        llm_b = CapturingLLM(prompts_b, SENTINEL_RESPONSES[:])
        Pipeline(passes=ALL_PASSES, backend=llm_a, style_profile=profile_a).run(INPUT_DRAFT)
        Pipeline(passes=ALL_PASSES, backend=llm_b, style_profile=profile_b).run(INPUT_DRAFT)

        assert prompts_a[3] != prompts_b[3], (
            "Different style profiles must produce different consistency-pass prompts"
        )

    def test_omitting_profile_does_not_break_pipeline(self) -> None:
        llm = MockLLM(SENTINEL_RESPONSES)
        results = Pipeline(passes=ALL_PASSES, backend=llm, style_profile=None).run(INPUT_DRAFT)
        assert len(results) == 5

    def test_pipeline_with_profile_and_without_produce_same_result_count(self) -> None:
        with_profile = StyleProfile.learn([SAMPLE_STYLE_TEXT])
        llm_a = MockLLM(SENTINEL_RESPONSES[:])
        llm_b = MockLLM(SENTINEL_RESPONSES[:])
        results_with = Pipeline(
            passes=ALL_PASSES, backend=llm_a, style_profile=with_profile
        ).run(INPUT_DRAFT)
        results_without = Pipeline(
            passes=ALL_PASSES, backend=llm_b, style_profile=None
        ).run(INPUT_DRAFT)
        assert len(results_with) == len(results_without)
