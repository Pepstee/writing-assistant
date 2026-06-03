from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field


_CONNECTORS = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "additionally", "meanwhile", "nonetheless", "thus",
    "hence", "although", "though", "whereas", "yet", "but", "also",
    "besides", "indeed", "accordingly",
}

_POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "positive", "happy", "joy", "love", "best", "perfect", "beautiful",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "negative", "sad", "hate",
    "worst", "ugly", "poor", "dreadful", "failure", "wrong",
}


def _tokenize_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class StyleProfile:
    avg_sentence_length: float = 0.0
    bigram_freq: Counter = field(default_factory=Counter)
    trigram_freq: Counter = field(default_factory=Counter)
    connector_freq: Counter = field(default_factory=Counter)
    tone_fingerprint: dict[str, float] = field(default_factory=dict)

    def fit(self, samples: list[str]) -> "StyleProfile":
        all_sentences: list[str] = []
        all_words: list[str] = []

        for text in samples:
            sentences = _tokenize_sentences(text)
            all_sentences.extend(sentences)
            words = _tokenize_words(text)
            all_words.extend(words)

        if all_sentences:
            lengths = [len(_tokenize_words(s)) for s in all_sentences]
            self.avg_sentence_length = sum(lengths) / len(lengths)

        self.bigram_freq = Counter(_ngrams(all_words, 2))
        self.trigram_freq = Counter(_ngrams(all_words, 3))

        self.connector_freq = Counter(w for w in all_words if w in _CONNECTORS)

        total = len(all_words) or 1
        pos_count = sum(1 for w in all_words if w in _POSITIVE_WORDS)
        neg_count = sum(1 for w in all_words if w in _NEGATIVE_WORDS)
        full_text = " ".join(samples)
        question_density = full_text.count("?") / total
        exclaim_density = full_text.count("!") / total

        self.tone_fingerprint = {
            "positivity": pos_count / total,
            "negativity": neg_count / total,
            "question_density": question_density,
            "exclaim_density": exclaim_density,
            "avg_word_length": (
                sum(len(w) for w in all_words) / total if all_words else 0.0
            ),
        }

        return self

    def similarity_score(self, text: str) -> float:
        if not self.tone_fingerprint:
            return 0.0

        words = _tokenize_words(text)
        sentences = _tokenize_sentences(text)

        # Sentence length similarity
        if sentences:
            lengths = [len(_tokenize_words(s)) for s in sentences]
            text_avg_sl = sum(lengths) / len(lengths)
        else:
            text_avg_sl = 0.0

        ref = self.avg_sentence_length
        sl_score = 1.0 - min(abs(text_avg_sl - ref) / max(ref, 1.0), 1.0)

        # N-gram similarity
        text_bigrams = Counter(_ngrams(words, 2))
        text_trigrams = Counter(_ngrams(words, 3))
        bigram_score = _cosine_similarity(self.bigram_freq, text_bigrams)
        trigram_score = _cosine_similarity(self.trigram_freq, text_trigrams)

        # Tone similarity
        total = len(words) or 1
        pos = sum(1 for w in words if w in _POSITIVE_WORDS) / total
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS) / total
        q_den = text.count("?") / total
        ex_den = text.count("!") / total
        avg_wl = sum(len(w) for w in words) / total if words else 0.0

        fp = self.tone_fingerprint
        tone_dims = [
            1.0 - min(abs(pos - fp["positivity"]) / max(fp["positivity"], 1e-9), 1.0),
            1.0 - min(abs(neg - fp["negativity"]) / max(fp["negativity"], 1e-9), 1.0),
            1.0 - min(abs(q_den - fp["question_density"]) / max(fp["question_density"], 1e-9), 1.0),
            1.0 - min(abs(ex_den - fp["exclaim_density"]) / max(fp["exclaim_density"], 1e-9), 1.0),
            1.0 - min(abs(avg_wl - fp["avg_word_length"]) / max(fp["avg_word_length"], 1.0), 1.0),
        ]
        tone_score = sum(tone_dims) / len(tone_dims)

        score = (
            0.30 * sl_score
            + 0.25 * bigram_score
            + 0.15 * trigram_score
            + 0.30 * tone_score
        )
        return max(0.0, min(1.0, score))
