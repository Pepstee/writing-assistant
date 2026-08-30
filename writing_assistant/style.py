from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TRANSITION_WORDS: frozenset[str] = frozenset({
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "additionally", "meanwhile", "nonetheless", "thus",
    "hence", "although", "though", "whereas", "yet", "also",
    "besides", "indeed", "accordingly", "finally", "similarly",
})

_PASSIVE_RE = re.compile(
    r"\b(?:was|were|is|are|been|being|be)\s+\w+ed\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class StyleProfile:
    avg_sentence_length: float = 0.0
    passive_voice_ratio: float = 0.0
    vocabulary_richness: float = 0.0
    preferred_transition_words: list[str] = field(default_factory=list)

    @classmethod
    def learn(cls, samples: list[str]) -> StyleProfile:
        all_sents: list[str] = []
        all_words: list[str] = []
        for text in samples:
            all_sents.extend(_sentences(text))
            all_words.extend(_words(text))

        if all_sents:
            avg_sl = sum(len(_words(s)) for s in all_sents) / len(all_sents)
            passive_ratio = sum(
                1 for s in all_sents if _PASSIVE_RE.search(s)
            ) / len(all_sents)
        else:
            avg_sl = 0.0
            passive_ratio = 0.0

        vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0.0

        joined = " ".join(samples).lower()
        transitions = sorted(
            w for w in _TRANSITION_WORDS
            if re.search(r"\b" + re.escape(w) + r"\b", joined)
        )

        return cls(
            avg_sentence_length=avg_sl,
            passive_voice_ratio=passive_ratio,
            vocabulary_richness=vocab_richness,
            preferred_transition_words=transitions,
        )

    def to_json(self) -> str:
        return json.dumps({
            "avg_sentence_length": self.avg_sentence_length,
            "passive_voice_ratio": self.passive_voice_ratio,
            "vocabulary_richness": self.vocabulary_richness,
            "preferred_transition_words": self.preferred_transition_words,
        })

    @classmethod
    def from_json(cls, s: str) -> StyleProfile:
        d = json.loads(s)
        return cls(
            avg_sentence_length=d["avg_sentence_length"],
            passive_voice_ratio=d["passive_voice_ratio"],
            vocabulary_richness=d["vocabulary_richness"],
            preferred_transition_words=d["preferred_transition_words"],
        )

    def summary(self) -> str:
        lines = [
            f"Average sentence length: {self.avg_sentence_length:.1f} words",
            f"Passive voice ratio: {self.passive_voice_ratio:.1%}",
            f"Vocabulary richness (type-token ratio): {self.vocabulary_richness:.2f}",
        ]
        if self.preferred_transition_words:
            lines.append(
                f"Preferred transition words: {', '.join(self.preferred_transition_words)}"
            )
        return "\n".join(lines)


_DESIRED_DEFAULTS: dict[str, Any] = {
    "tone": "neutral",
    "formality": "informal",
    "vocabulary": [],
    "max_sentence_words": 30,
    "min_sentence_words": 5,
}
_VALID_TONES = {"neutral", "formal", "friendly", "assertive", "empathetic"}
_VALID_FORMALITIES = {"formal", "semiformal", "informal"}


@dataclass
class DesiredStyleProfile:
    """Validated, operator-authored style guidance loaded from JSON or TOML.

    This is deliberately distinct from :class:`StyleProfile`, which measures
    observed writing samples.  A desired profile states how a rewrite should
    read; it does not pretend those preferences were learned from source text.
    """

    tone: str = "neutral"
    formality: str = "informal"
    vocabulary: list[str] = field(default_factory=list)
    max_sentence_words: int = 30
    min_sentence_words: int = 5

    def __post_init__(self) -> None:
        if self.tone not in _VALID_TONES:
            raise ValueError(
                f"Invalid tone {self.tone!r}. Must be one of: "
                f"{', '.join(sorted(_VALID_TONES))}."
            )
        if self.formality not in _VALID_FORMALITIES:
            raise ValueError(
                f"Invalid formality {self.formality!r}. Must be one of: "
                f"{', '.join(sorted(_VALID_FORMALITIES))}."
            )
        if not isinstance(self.vocabulary, list) or not all(
            isinstance(word, str) and word for word in self.vocabulary
        ):
            raise ValueError("vocabulary must be a list of non-empty strings.")
        if type(self.max_sentence_words) is not int or self.max_sentence_words < 1:
            raise ValueError("max_sentence_words must be a positive integer.")
        if type(self.min_sentence_words) is not int or self.min_sentence_words < 1:
            raise ValueError("min_sentence_words must be a positive integer.")
        if self.min_sentence_words > self.max_sentence_words:
            raise ValueError(
                f"min_sentence_words ({self.min_sentence_words}) must not exceed "
                f"max_sentence_words ({self.max_sentence_words})."
            )

    @classmethod
    def from_file(cls, path: str | Path) -> DesiredStyleProfile:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Profile file not found: {source}")
        suffix = source.suffix.lower()
        if suffix == ".json":
            data = cls._load_json(source)
        elif suffix == ".toml":
            data = cls._load_toml(source)
        else:
            raise ValueError(
                f"Unsupported profile format {suffix!r}. Use .json or .toml."
            )
        unknown = set(data) - set(_DESIRED_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown fields in {source}: {', '.join(sorted(unknown))}."
            )
        merged = {**_DESIRED_DEFAULTS, **data}
        if isinstance(merged["vocabulary"], list):
            merged["vocabulary"] = list(merged["vocabulary"])
        try:
            return cls(**merged)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid profile in {source}: {exc}") from exc

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON profile {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"JSON profile {path} must be a top-level object.")
        return data

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        if sys.version_info >= (3, 11):
            import tomllib

            try:
                with path.open("rb") as handle:
                    data = tomllib.load(handle)
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"Failed to parse TOML profile {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"TOML profile {path} must be a top-level table.")
            return data
        return _parse_simple_toml(path)

    def summary(self) -> str:
        self.__post_init__()
        vocabulary = ", ".join(self.vocabulary) if self.vocabulary else "none specified"
        return "\n".join(
            (
                f"Desired tone: {self.tone}",
                f"Desired formality: {self.formality}",
                f"Preferred vocabulary: {vocabulary}",
                "Sentence length: "
                f"{self.min_sentence_words}-{self.max_sentence_words} words",
            )
        )


def _parse_simple_toml(path: Path) -> dict[str, Any]:
    """Parse the profile's small top-level TOML subset on Python 3.10."""

    data: dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(
                f"Syntax error in {path} at line {line_number}: expected 'key = value'."
            )
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid key {key!r} in {path} at line {line_number}.")
        if key in data:
            raise ValueError(f"Duplicate key {key!r} in {path} at line {line_number}.")
        data[key] = _parse_toml_value(raw_value.strip(), path, line_number)
    return data


def _parse_toml_value(raw: str, path: Path, line_number: int) -> Any:
    if raw.startswith("["):
        if not raw.endswith("]"):
            raise ValueError(f"Unterminated array in {path} at line {line_number}.")
        inner = raw[1:-1].strip()
        if not inner:
            return []
        values: list[str] = []
        for item in re.split(r",\s*", inner):
            item = item.strip()
            if len(item) >= 2 and item[0] == item[-1] and item[0] in {'"', "'"}:
                values.append(item[1:-1])
            else:
                raise ValueError(
                    f"Array items must be quoted strings in {path} "
                    f"at line {line_number}."
                )
        return values
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    raise ValueError(
        f"Cannot parse value {raw!r} in {path} at line {line_number}."
    )
