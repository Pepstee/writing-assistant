from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


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
