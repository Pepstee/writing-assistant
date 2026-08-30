"""CLI entry point: python -m writing_assistant [options] [file]"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.style import StyleProfile
from writing_assistant.types import Pass, RewriteResult

_PASS_MAP = {
    "clarity": CLARITY,
    "tone": TONE,
    "conciseness": CONCISENESS,
    "consistency": CONSISTENCY,
    "adversarial": ADVERSARIAL,
}

_DEFAULT_PASSES = ["clarity", "tone", "conciseness", "consistency", "adversarial"]


def _render_console(
    passes: list[Pass], results: list[RewriteResult], style_summary: str | None
) -> str:
    """Render the existing human-readable CLI output byte-for-byte."""

    output = io.StringIO()
    if style_summary is not None:
        print(style_summary, file=output)
        print(file=output)
    for rewrite_pass, result in zip(passes, results):
        print(f"\n{'=' * 60}", file=output)
        print(f"Pass: {rewrite_pass.name.upper()}", file=output)
        print("=" * 60, file=output)
        if rewrite_pass.metadata.get("adversarial"):
            print(
                "Adversarial review: identifying weaknesses and rewriting to address them.",
                file=output,
            )
        print(result.diff if result.diff else "(no changes)", file=output)

    print(f"\n{'=' * 60}", file=output)
    print("Final draft:", file=output)
    print("=" * 60, file=output)
    print(results[-1].revised, file=output)
    return output.getvalue()


def _render_markdown(
    passes: list[Pass], results: list[RewriteResult], style_summary: str | None
) -> str:
    """Render the final text and every existing per-pass diff as Markdown."""

    lines: list[str] = []
    if style_summary is not None:
        lines.extend(("## Style Profile", "", style_summary, ""))
    lines.extend(("## Final Text", "", "```", results[-1].revised.rstrip("\n"), "```", ""))
    lines.extend(("## Pass Diffs", ""))
    for rewrite_pass, result in zip(passes, results):
        lines.extend(
            (
                f"### {rewrite_pass.name}",
                "",
                "```diff",
                result.diff.rstrip("\n") if result.diff else "(no changes)",
                "```",
                "",
            )
        )
    return "\n".join(lines)


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
        "--backend",
        choices=("claude", "rules"),
        default="claude",
        help=(
            "rewrite backend: 'claude' shells out to the authenticated Claude CLI; "
            "'rules' is the deterministic offline rule-based editor (no network)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("console", "plain", "markdown"),
        default="console",
        help=(
            "output format: 'console' preserves the existing human-readable "
            "report; 'plain' emits only the final text; 'markdown' emits the "
            "final text and per-pass diffs"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="write the complete rendered result to FILE (omit or use '-' for stdout)",
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
    if not pass_names:
        parser.error("--passes must contain at least one pass name")
    unknown = [p for p in pass_names if p not in _PASS_MAP]
    if unknown:
        parser.error(f"unknown pass(es): {', '.join(unknown)}. Available: {', '.join(_PASS_MAP)}")
    passes = [_PASS_MAP[n] for n in pass_names]

    # Optional style profile
    style_profile: StyleProfile | None = None
    style_summary: str | None = None
    if args.sample:
        try:
            with open(args.sample) as fh:
                sample_text = fh.read()
        except OSError as exc:
            parser.error(str(exc))
        style_profile = StyleProfile.learn([sample_text])
        style_summary = f"Style profile learned from {args.sample!r}:\n{style_profile.summary()}"

    # Build backend
    if args.backend == "rules":
        from writing_assistant.llm.rule_based import RuleBasedRewriter

        backend: object = RuleBasedRewriter()
    else:
        from writing_assistant.llm.claude_cli import ClaudeCliLLM

        backend = ClaudeCliLLM()

    # Run
    pipeline = Pipeline(passes=passes, backend=backend, style_profile=style_profile)
    results = pipeline.run(text)

    if args.format == "markdown":
        rendered = _render_markdown(passes, results, style_summary)
    elif args.format == "plain":
        rendered = results[-1].revised
    else:
        rendered = _render_console(passes, results, style_summary)
    try:
        if args.output is None or args.output == "-":
            sys.stdout.write(rendered)
            if not rendered.endswith("\n"):
                sys.stdout.write("\n")
        else:
            Path(args.output).write_text(rendered, encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"writing-assistant: error writing output: {exc}\n")


if __name__ == "__main__":
    main()
