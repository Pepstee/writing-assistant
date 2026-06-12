"""Adversarial tests for the two built-in LLM backends.

MockLLM is tested directly (it is fully deterministic). ClaudeCliLLM shells
out to the `claude` CLI, so subprocess.run — the external boundary — is
patched; everything on our side of that boundary is exercised for real.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from writing_assistant.llm.claude_cli import ClaudeCliLLM
from writing_assistant.llm.mock import MockLLM
from writing_assistant.types import LLMBackend


# ── MockLLM: list mode ─────────────────────────────────────────────────────────

class TestMockLLMListMode:
    def test_returns_configured_response(self):
        assert MockLLM(["only"]).generate("anything") == "only"

    def test_cycles_through_responses_in_order(self):
        llm = MockLLM(["a", "b", "c"])
        assert [llm.generate("p") for _ in range(3)] == ["a", "b", "c"]

    def test_wraps_around_after_last_response(self):
        llm = MockLLM(["a", "b"])
        assert [llm.generate("p") for _ in range(5)] == ["a", "b", "a", "b", "a"]

    def test_single_response_repeats_indefinitely(self):
        llm = MockLLM(["same"])
        assert [llm.generate("p") for _ in range(4)] == ["same"] * 4

    def test_empty_list_falls_back_to_default(self):
        assert MockLLM([]).generate("p") == "mock response"

    def test_prompt_content_does_not_affect_response(self):
        a, b = MockLLM(["x"]), MockLLM(["x"])
        assert a.generate("short") == b.generate("a completely different prompt")

    def test_instances_have_independent_state(self):
        a, b = MockLLM(["1", "2"]), MockLLM(["1", "2"])
        a.generate("p")
        assert b.generate("p") == "1"


# ── MockLLM: dict mode and defaults ────────────────────────────────────────────

class TestMockLLMDictMode:
    def test_exact_prompt_lookup(self):
        llm = MockLLM({"What?": "That.", "Who?": "Them."})
        assert llm.generate("What?") == "That."
        assert llm.generate("Who?") == "Them."

    def test_unknown_prompt_returns_default(self):
        assert MockLLM({"known": "value"}).generate("unknown") == "mock response"

    def test_no_argument_returns_default(self):
        assert MockLLM().generate("anything") == "mock response"

    def test_dict_lookup_is_exact_not_substring(self):
        llm = MockLLM({"prompt": "matched"})
        assert llm.generate("prompt with suffix") == "mock response"

    def test_satisfies_llm_backend_protocol(self):
        backend: LLMBackend = MockLLM()
        assert backend.generate("p") == "mock response"


# ── ClaudeCliLLM ───────────────────────────────────────────────────────────────

def _completed(stdout: str) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    return proc


class TestClaudeCliLLM:
    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_invokes_claude_cli_with_prompt(self, run):
        run.return_value = _completed("out")
        ClaudeCliLLM().generate("my prompt")
        cmd = run.call_args.args[0]
        assert cmd[0] == "claude"
        assert cmd[-2:] == ["-p", "my prompt"]

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_default_model_passed_via_flag(self, run):
        run.return_value = _completed("out")
        ClaudeCliLLM().generate("p")
        cmd = run.call_args.args[0]
        assert cmd[1:3] == ["--model", "claude-sonnet-4-6"]

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_custom_model_passed_via_flag(self, run):
        run.return_value = _completed("out")
        ClaudeCliLLM(model="claude-haiku-4-5-20251001").generate("p")
        assert run.call_args.args[0][1:3] == ["--model", "claude-haiku-4-5-20251001"]

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_extra_args_inserted_before_prompt(self, run):
        run.return_value = _completed("out")
        ClaudeCliLLM(extra_args=["--max-turns", "1"]).generate("p")
        cmd = run.call_args.args[0]
        assert cmd[3:5] == ["--max-turns", "1"]
        assert cmd[-2:] == ["-p", "p"]

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_returns_stripped_stdout(self, run):
        run.return_value = _completed("  rewritten text \n")
        assert ClaudeCliLLM().generate("p") == "rewritten text"

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_runs_with_check_capture_and_text(self, run):
        run.return_value = _completed("out")
        ClaudeCliLLM().generate("p")
        kwargs = run.call_args.kwargs
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    @patch("writing_assistant.llm.claude_cli.subprocess.run")
    def test_cli_failure_propagates(self, run):
        run.side_effect = subprocess.CalledProcessError(1, ["claude"])
        with pytest.raises(subprocess.CalledProcessError):
            ClaudeCliLLM().generate("p")

    def test_satisfies_llm_backend_protocol(self):
        backend: LLMBackend = ClaudeCliLLM()
        assert callable(backend.generate)
