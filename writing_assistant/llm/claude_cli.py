from __future__ import annotations

import subprocess
from collections.abc import Sequence

from writing_assistant.types import LLMBackend


class LLMBackendError(RuntimeError):
    """A configured command backend could not return usable rewrite text."""


class ClaudeCliLLM:
    """LLM backend that shells out to the authenticated `claude` CLI."""

    def __init__(
        self, model: str = "claude-sonnet-4-6", extra_args: list[str] | None = None
    ) -> None:
        self.model = model
        self.extra_args = extra_args or []

    def generate(self, prompt: str) -> str:
        cmd = ["claude", "--model", self.model, *self.extra_args, "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()


class CommandCliLLM:
    """Model-agnostic CLI backend that exchanges prompts and rewrites over stdio.

    The command is always executed as a direct argument vector, never through a
    shell. The exact prompt is written to stdin and stripped stdout is returned.
    """

    def __init__(self, command: Sequence[str], *, timeout: float = 120) -> None:
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("command must be a non-empty argument sequence")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if not all(isinstance(argument, str) for argument in command):
            raise TypeError("every command argument must be a string")
        if not command[0]:
            raise ValueError("command executable must not be empty")
        self.command = list(command)
        self.timeout = float(timeout)

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                self.command,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise LLMBackendError(f"command not found: {self.command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMBackendError(
                f"command timed out after {self.timeout:g} seconds: {self.command[0]}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-400:]
            suffix = f": {detail}" if detail else ""
            raise LLMBackendError(
                f"command exited {exc.returncode}: {self.command[0]}{suffix}"
            ) from exc
        except UnicodeError as exc:
            raise LLMBackendError(
                f"command returned non-UTF-8 output: {self.command[0]}"
            ) from exc
        except OSError as exc:
            raise LLMBackendError(f"command failed: {self.command[0]}: {exc}") from exc

        output = result.stdout.strip()
        if not output:
            raise LLMBackendError(f"command returned no output: {self.command[0]}")
        return output


_: LLMBackend = ClaudeCliLLM()
