"""Deterministic, offline rewrite backend.

``RuleBasedRewriter`` is not a language model: it is a rule engine that
implements the same :class:`~writing_assistant.types.LLMBackend` protocol.
It detects which pipeline pass is asking (from the pass instructions embedded
in the prompt), extracts the text payload, and applies that pass's editing
rules — plain-language substitutions for clarity, filler removal for
conciseness, contraction expansion and register fixes for tone, typographic
normalisation for consistency, and the full rule set for the adversarial
review. Dependency-free and fully offline, it lets the product demonstrate
itself end-to-end (CLI ``--backend rules`` and the acceptance demo) without a
network or an API key, and it is honest about what it is: every edit it makes
is a real, inspectable transformation.
"""
from __future__ import annotations

import re

from writing_assistant.types import LLMBackend

# ── Clarity: wordy phrase → plain language ────────────────────────────────────
# Ordered longest-first so broader phrases win before their substrings.
_CLARITY_PHRASES: list[tuple[str, str]] = [
    ("with the explicit purpose of ensuring that", "so that"),
    ("take into consideration the fact that", "consider that"),
    ("it is of the utmost importance to", "it is important to"),
    ("in spite of the fact that", "although"),
    ("due to the fact that", "because"),
    ("take into consideration", "consider"),
    ("making utilization of", "using"),
    ("make utilization of", "use"),
    ("for the purpose of", "to"),
    ("at this point in time", "now"),
    ("at any point in time", "at any time"),
    ("in the event that", "if"),
    ("the aforementioned", "these"),
    ("aforementioned", "these"),
    ("a large number of", "many"),
    ("a number of", "several"),
    ("in order to", "to"),
    ("utilization", "use"),
    ("utilizes", "uses"),
    ("utilized", "used"),
    ("utilize", "use"),
    ("facilitates", "helps with"),
    ("facilitated", "helped with"),
    ("facilitate", "help with"),
    ("numerous", "many"),
    ("commence", "begin"),
    ("endeavour", "attempt"),
    ("endeavor", "attempt"),
]

# ── Conciseness: filler that adds words but no meaning ────────────────────────
_CONCISENESS_PHRASES: list[tuple[str, str]] = [
    ("it should furthermore be noted that", ""),
    ("it should be noted that", ""),
    ("it is worth noting that", ""),
    ("needless to say,", ""),
    ("include, but are not limited to,", "include"),
    ("includes, but is not limited to,", "includes"),
    ("across the entirety of", "across"),
    ("the entirety of", "all of"),
    ("regardless of their level of expertise or experience", "at any skill level"),
    ("at the present moment", "now"),
    ("absolutely essential", "essential"),
    ("completely eliminate", "eliminate"),
    ("first and foremost", "first"),
    ("each and every", "every"),
    ("in a meaningful way", ""),
    ("basically", ""),
    ("actually", ""),
    ("really", ""),
    ("very", ""),
]

# ── Tone: contractions and informal register → professional ──────────────────
_TONE_PHRASES: list[tuple[str, str]] = [
    ("can't", "cannot"),
    ("won't", "will not"),
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("didn't", "did not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("it's", "it is"),
    ("we're", "we are"),
    ("you're", "you are"),
    ("they're", "they are"),
    ("gonna", "going to"),
    ("wanna", "want to"),
    ("a lot of", "many"),
    ("lots of", "many"),
    ("awesome", "excellent"),
    ("stuff", "material"),
]

# ── Consistency: typographic / terminology normalisation ─────────────────────
_CONSISTENCY_PHRASES: list[tuple[str, str]] = [
    ("e-mail", "email"),
    ("web site", "website"),
    ("web-site", "website"),
]
_CURLY_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}

# Pass detection: a distinctive marker from each built-in pass's instructions.
_PASS_MARKERS: list[tuple[str, str]] = [
    ("adversarial editor", "adversarial"),
    ("improve clarity", "clarity"),
    ("respectful tone", "tone"),
    ("more concise", "conciseness"),
    ("consistent terminology", "consistency"),
]


def _replace_phrases(text: str, phrases: list[tuple[str, str]]) -> str:
    """Apply word-boundary, case-insensitive substitutions, preserving a
    leading capital on the replacement when the original started one."""
    for old, new in phrases:
        tail = r"\b" if old[-1].isalnum() else ""
        pattern = re.compile(r"\b" + re.escape(old) + tail, re.IGNORECASE)

        def _sub(m: re.Match[str], new: str = new) -> str:
            if new and m.group(0)[0].isupper():
                return new[0].upper() + new[1:]
            return new

        text = pattern.sub(_sub, text)
    return text


def _tidy(text: str) -> str:
    """Repair artefacts left by deletions: stray spaces, orphaned punctuation,
    and lowercase sentence openers."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?,;:]) *([.!?,;:])", r"\1\2", text)
    text = re.sub(
        r"(^\s*|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    return text.strip()


def _clarity(text: str) -> str:
    return _tidy(_replace_phrases(text, _CLARITY_PHRASES))


def _conciseness(text: str) -> str:
    return _tidy(_replace_phrases(text, _CONCISENESS_PHRASES))


def _tone(text: str) -> str:
    text = _replace_phrases(text, _TONE_PHRASES)
    text = text.replace("!", ".")
    return _tidy(text)


def _consistency(text: str) -> str:
    for curly, straight in _CURLY_QUOTES.items():
        text = text.replace(curly, straight)
    text = text.replace("--", "—")
    text = _replace_phrases(text, _CONSISTENCY_PHRASES)
    return _tidy(text)


def _adversarial(text: str) -> str:
    """The adversarial review re-applies every rule set; if nothing fires,
    the text survives review unchanged."""
    for fn in (_clarity, _conciseness, _tone, _consistency):
        text = fn(text)
    return text


_PASS_FNS = {
    "clarity": _clarity,
    "tone": _tone,
    "conciseness": _conciseness,
    "consistency": _consistency,
    "adversarial": _adversarial,
}


class RuleBasedRewriter:
    """Offline ``LLMBackend``: rule-based editing instead of model inference."""

    def generate(self, prompt: str) -> str:
        instructions, text = self._split(prompt)
        lowered = instructions.lower()
        for marker, name in _PASS_MARKERS:
            if marker in lowered:
                return _PASS_FNS[name](text)
        # Unknown instructions: run the full editing battery.
        return _adversarial(text)

    @staticmethod
    def _split(prompt: str) -> tuple[str, str]:
        """Separate the pipeline's instruction header from the text payload."""
        for marker in ("\nCurrent text:\n", "\nText:\n"):
            if marker in prompt:
                head, _, tail = prompt.partition(marker)
                return head, tail
        return prompt, prompt


# Satisfy the Protocol at definition time.
_: LLMBackend = RuleBasedRewriter()
