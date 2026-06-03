"""
Adversarial test suite for writing_assistant.backends.

Coverage:
- MockBackend: construction, cycling, wrap-around, prompt independence, instance isolation
- ClaudeCLIBackend: command construction, subprocess isolation, stdout stripping,
  error propagation, extra_args threading
- Protocol conformance: both backends satisfy the complete(str) -> str contract
- Mutation-resistant: tight invariants that would break on common implementation drift
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from writing_assistant.backends import ClaudeCLIBackend, MockBackend


# ── MockBackend construction ───────────────────────────────────────────────────

class TestMockBackendConstruction:
    def test_no_args_uses_default_response(self):
        backend = MockBackend()
        assert backend.complete("anything") == "mock response"

    def test_none_arg_uses_default_response(self):
        backend = MockBackend(None)
        assert backend.complete("anything") == "mock response"

    def test_empty_list_falls_back_to_default(self):
        # [] is falsy; the `responses or ["mock response"]` branch fires
        backend = MockBackend([])
        assert backend.complete("p") == "mock response"

    def test_single_response_list_stored(self):
        backend = MockBackend(["custom"])
        assert backend.complete("p") == "custom"

    def test_multiple_response_list_stored_in_order(self):
        backend = MockBackend(["first", "second"])
        assert backend.complete("p") == "first"

    def test_responses_list_not_mutated_externally(self):
        responses = ["alpha", "beta"]
        backend = MockBackend(responses)
        responses.clear()
        # backend should still function with its own copy/deque
        result = backend.complete("p")
        assert result in ("alpha", "beta")


# ── MockBackend.complete() cycling and wrap-around ────────────────────────────

class TestMockBackendComplete:
    def test_single_response_returned_every_call(self):
        backend = MockBackend(["only"])
        for _ in range(5):
            assert backend.complete("p") == "only"

    def test_two_responses_cycle_in_order(self):
        backend = MockBackend(["a", "b"])
        assert backend.complete("p") == "a"
        assert backend.complete("p") == "b"

    def test_two_responses_wrap_after_second(self):
        backend = MockBackend(["a", "b"])
        backend.complete("p")  # a
        backend.complete("p")  # b
        assert backend.complete("p") == "a"  # back to start

    def test_three_responses_full_cycle(self):
        backend = MockBackend(["x", "y", "z"])
        assert backend.complete("p") == "x"
        assert backend.complete("p") == "y"
        assert backend.complete("p") == "z"
        assert backend.complete("p") == "x"  # wrap

    def test_cycle_is_deterministic_across_many_calls(self):
        backend = MockBackend(["r0", "r1", "r2"])
        results = [backend.complete("p") for _ in range(9)]
        assert results == ["r0", "r1", "r2", "r0", "r1", "r2", "r0", "r1", "r2"]

    def test_prompt_content_does_not_affect_response(self):
        backend = MockBackend(["fixed"])
        assert backend.complete("") == "fixed"
        assert backend.complete("a" * 10_000) == "fixed"
        assert backend.complete("completely different") == "fixed"

    def test_return_type_is_str(self):
        backend = MockBackend(["hello"])
        assert isinstance(backend.complete("p"), str)

    def test_response_is_not_the_prompt(self):
        prompt = "the prompt"
        backend = MockBackend(["the response"])
        result = backend.complete(prompt)
        assert result != prompt

    def test_empty_string_response_returned_verbatim(self):
        backend = MockBackend([""])
        assert backend.complete("p") == ""

    def test_whitespace_response_returned_verbatim(self):
        backend = MockBackend(["   \n  "])
        assert backend.complete("p") == "   \n  "

    def test_multiline_response_returned_intact(self):
        multiline = "line one\nline two\nline three"
        backend = MockBackend([multiline])
        assert backend.complete("p") == multiline


# ── MockBackend instance isolation ────────────────────────────────────────────

class TestMockBackendInstanceIsolation:
    def test_two_instances_advance_independently(self):
        b1 = MockBackend(["a", "b"])
        b2 = MockBackend(["x", "y"])
        assert b1.complete("p") == "a"
        assert b2.complete("p") == "x"
        assert b1.complete("p") == "b"
        assert b2.complete("p") == "y"

    def test_advancing_one_instance_does_not_affect_other(self):
        b1 = MockBackend(["only_one"])
        b2 = MockBackend(["also_one"])
        for _ in range(10):
            b1.complete("p")
        assert b2.complete("p") == "also_one"

    def test_instances_with_same_responses_have_separate_state(self):
        b1 = MockBackend(["shared", "different"])
        b2 = MockBackend(["shared", "different"])
        b1.complete("p")  # advance b1 to "different"
        # b2 should still be at "shared"
        assert b2.complete("p") == "shared"


# ── ClaudeCLIBackend construction ─────────────────────────────────────────────

class TestClaudeCLIBackendConstruction:
    def test_default_model_is_claude_sonnet(self):
        backend = ClaudeCLIBackend()
        assert backend.model == "claude-sonnet-4-6"

    def test_custom_model_stored(self):
        backend = ClaudeCLIBackend(model="claude-opus-4-8")
        assert backend.model == "claude-opus-4-8"

    def test_default_extra_args_is_empty_list(self):
        backend = ClaudeCLIBackend()
        assert backend.extra_args == []

    def test_none_extra_args_becomes_empty_list(self):
        backend = ClaudeCLIBackend(extra_args=None)
        assert backend.extra_args == []

    def test_custom_extra_args_stored(self):
        backend = ClaudeCLIBackend(extra_args=["--verbose"])
        assert backend.extra_args == ["--verbose"]

    def test_multiple_extra_args_stored_in_order(self):
        args = ["--flag", "value", "--other"]
        backend = ClaudeCLIBackend(extra_args=args)
        assert backend.extra_args == args

    def test_model_and_extra_args_independent(self):
        backend = ClaudeCLIBackend(model="my-model", extra_args=["--x"])
        assert backend.model == "my-model"
        assert backend.extra_args == ["--x"]


# ── ClaudeCLIBackend subprocess isolation ─────────────────────────────────────

class TestClaudeCLIBackendSubprocessIsolation:
    """subprocess.run is the external system boundary — patching it is correct here."""

    def _make_result(self, stdout: str = "response", returncode: int = 0) -> MagicMock:
        m = MagicMock()
        m.stdout = stdout
        m.returncode = returncode
        return m

    def test_complete_calls_subprocess_run_once(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("hi")
        mock_run.assert_called_once()

    def test_command_starts_with_claude(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("hi")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"

    def test_command_includes_model_flag(self):
        backend = ClaudeCLIBackend(model="claude-haiku-4-5")
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-haiku-4-5"

    def test_command_includes_prompt_flag(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("my prompt text")
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        p_idx = cmd.index("-p")
        assert cmd[p_idx + 1] == "my prompt text"

    def test_command_includes_extra_args(self):
        backend = ClaudeCLIBackend(extra_args=["--dangerously-skip-permissions"])
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        assert "--dangerously-skip-permissions" in cmd

    def test_extra_args_appear_before_prompt_flag(self):
        backend = ClaudeCLIBackend(extra_args=["--extra"])
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        extra_idx = cmd.index("--extra")
        p_idx = cmd.index("-p")
        assert extra_idx < p_idx

    def test_multiple_extra_args_all_present_in_order(self):
        backend = ClaudeCLIBackend(extra_args=["--flag-a", "val-a", "--flag-b"])
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("out")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        a_idx = cmd.index("--flag-a")
        b_idx = cmd.index("--flag-b")
        assert a_idx < b_idx
        assert cmd[a_idx + 1] == "val-a"

    def test_capture_output_true(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result()
            backend.complete("p")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("capture_output") is True

    def test_text_mode_true(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result()
            backend.complete("p")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text") is True

    def test_check_true_so_nonzero_exit_raises(self):
        """check=True means subprocess.run raises on non-zero exit — verify it propagates."""
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result()
            backend.complete("p")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is True

    def test_complete_returns_stdout_stripped(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("result text\n  ")
            result = backend.complete("p")
        assert result == "result text"

    def test_complete_return_type_is_str(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("answer")
            result = backend.complete("p")
        assert isinstance(result, str)

    def test_stdout_with_no_whitespace_returned_unchanged(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("clean")
            result = backend.complete("p")
        assert result == "clean"

    def test_empty_stdout_returns_empty_string(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("")
            result = backend.complete("p")
        assert result == ""

    def test_stdout_only_whitespace_returns_empty_string(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("   \n\t  ")
            result = backend.complete("p")
        assert result == ""

    def test_called_process_error_propagates(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "claude")
            with pytest.raises(subprocess.CalledProcessError):
                backend.complete("p")

    def test_different_prompts_produce_different_commands(self):
        backend = ClaudeCLIBackend()
        calls_made = []

        def recording_run(cmd, **kwargs):
            calls_made.append(list(cmd))
            m = MagicMock()
            m.stdout = "ok"
            return m

        with patch("writing_assistant.backends.subprocess.run", side_effect=recording_run):
            backend.complete("prompt one")
            backend.complete("prompt two")

        assert calls_made[0] != calls_made[1]
        p_idx = calls_made[0].index("-p")
        assert calls_made[0][p_idx + 1] == "prompt one"
        assert calls_made[1][p_idx + 1] == "prompt two"

    def test_full_command_structure(self):
        """Exact command vector: ['claude', '--model', M, *extra_args, '-p', prompt]."""
        backend = ClaudeCLIBackend(model="mymodel", extra_args=["--ea"])
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("resp")
            backend.complete("myprompt")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["claude", "--model", "mymodel", "--ea", "-p", "myprompt"]

    def test_no_extra_args_command_structure(self):
        """Without extra_args, command is ['claude', '--model', M, '-p', prompt]."""
        backend = ClaudeCLIBackend(model="themodel")
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = self._make_result("r")
            backend.complete("theprompt")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["claude", "--model", "themodel", "-p", "theprompt"]


# ── Protocol conformance ───────────────────────────────────────────────────────

class TestBackendProtocolConformance:
    """Both backends must satisfy complete(str) -> str unconditionally."""

    @pytest.mark.parametrize("backend", [
        MockBackend(["response"]),
        MockBackend(),
    ])
    def test_mock_backend_complete_accepts_string(self, backend):
        result = backend.complete("prompt")
        assert isinstance(result, str)

    def test_mock_backend_complete_signature_takes_prompt(self):
        backend = MockBackend(["ok"])
        result = backend.complete(prompt="positional or keyword")
        assert isinstance(result, str)

    def test_claude_cli_backend_complete_returns_string(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="text result\n")
            result = backend.complete("some prompt")
        assert isinstance(result, str)

    def test_both_backends_are_usable_interchangeably_by_pass(self):
        """A pass must not distinguish between backends — duck typing contract."""
        from writing_assistant.passes import ClarityPass

        mock_backend = MockBackend(["improved text"])
        result = ClarityPass().run("original text", mock_backend)
        assert isinstance(result.rewritten, str)
        assert result.rewritten == "improved text"

    def test_mock_backend_has_complete_callable(self):
        assert callable(getattr(MockBackend(), "complete", None))

    def test_claude_cli_backend_has_complete_callable(self):
        assert callable(getattr(ClaudeCLIBackend(), "complete", None))

    def test_mock_backend_complete_never_returns_none(self):
        backend = MockBackend(["x"])
        assert backend.complete("p") is not None

    def test_claude_cli_backend_complete_never_returns_none(self):
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = backend.complete("p")
        assert result is not None


# ── Mutation-resistant edge cases ─────────────────────────────────────────────

class TestMutationResistant:
    def test_mock_backend_cycle_is_not_random(self):
        """Responses must cycle in the same order every time — not shuffled."""
        backend = MockBackend(["A", "B", "C"])
        first_cycle = [backend.complete("p") for _ in range(3)]
        second_cycle = [backend.complete("p") for _ in range(3)]
        assert first_cycle == second_cycle

    def test_mock_backend_does_not_pop_responses(self):
        """Responses are cycled, not consumed — they remain available after N calls."""
        backend = MockBackend(["only"])
        for _ in range(100):
            result = backend.complete("p")
            assert result == "only", "Response was consumed rather than cycled"

    def test_mock_backend_rotate_not_pop_on_multi_response(self):
        """After cycling all responses once, they all appear again in the same order."""
        backend = MockBackend(["p", "q"])
        seen = [backend.complete("x") for _ in range(4)]
        assert seen == ["p", "q", "p", "q"]

    def test_claude_cli_stdout_not_returned_raw(self):
        """stdout with trailing newline must be stripped — not returned raw."""
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="answer\n")
            result = backend.complete("p")
        assert result == "answer"
        assert not result.endswith("\n")

    def test_claude_cli_does_not_return_stderr(self):
        """Only stdout is part of the response — stderr must not bleed into output."""
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            m = MagicMock()
            m.stdout = "clean output"
            m.stderr = "some warning"
            mock_run.return_value = m
            result = backend.complete("p")
        assert "some warning" not in result

    def test_claude_cli_model_not_hardcoded_in_command(self):
        """A different model must reach the subprocess command unchanged."""
        backend = ClaudeCLIBackend(model="unusual-model-xyz")
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="r")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        assert "unusual-model-xyz" in cmd

    def test_claude_cli_extra_args_not_ignored(self):
        """Extra args must appear in the command — not silently dropped."""
        backend = ClaudeCLIBackend(extra_args=["--sentinel-arg"])
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="r")
            backend.complete("p")
        cmd = mock_run.call_args[0][0]
        assert "--sentinel-arg" in cmd

    def test_mock_backend_prompt_not_echoed_back(self):
        """MockBackend must not echo the prompt — return the configured response only."""
        prompt = "UNIQUE_PROMPT_NOT_IN_RESPONSE"
        backend = MockBackend(["configured_response"])
        result = backend.complete(prompt)
        assert result == "configured_response"
        assert prompt not in result

    def test_claude_cli_calledprocesserror_not_swallowed(self):
        """A non-zero subprocess exit must raise, not return an empty string."""
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(2, "claude", output="", stderr="err")
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                backend.complete("p")
        assert exc_info.value.returncode == 2

    def test_mock_backend_empty_string_response_not_replaced_with_default(self):
        """An explicit empty-string response must be returned as-is, not overridden."""
        backend = MockBackend([""])
        # The list is truthy (non-empty), so the default branch should NOT fire
        result = backend.complete("p")
        assert result == ""

    def test_mock_backend_wraps_not_raises_on_exhaustion(self):
        """After the last response, the next call must wrap (not IndexError)."""
        backend = MockBackend(["one", "two"])
        for _ in range(6):  # 3 full cycles
            backend.complete("p")  # must not raise

    def test_claude_cli_prompt_not_truncated(self):
        """Long prompt must be passed to subprocess verbatim — not truncated."""
        long_prompt = "word " * 1_000
        backend = ClaudeCLIBackend()
        with patch("writing_assistant.backends.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="r")
            backend.complete(long_prompt)
        cmd = mock_run.call_args[0][0]
        p_idx = cmd.index("-p")
        assert cmd[p_idx + 1] == long_prompt
