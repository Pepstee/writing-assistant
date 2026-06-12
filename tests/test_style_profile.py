"""
Adversarial pytest suite for writing_assistant.style_profile.

Rules:
- Never mock the unit under test — every assertion can fail if the implementation is wrong.
- Covers: happy paths, edge cases, boundary values, mutation resistance.
- Tests can FAIL; no trivially-true assertions.
"""
from __future__ import annotations

from collections import Counter

import pytest

from writing_assistant.style_profile import (
    StyleProfile,
    _cosine_similarity,
    _ngrams,
    _tokenize_sentences,
    _tokenize_words,
)


# ── _tokenize_sentences ───────────────────────────────────────────────────────


class TestTokenizeSentences:
    def test_empty_string_returns_empty_list(self):
        assert _tokenize_sentences("") == []

    def test_single_sentence_no_terminator(self):
        result = _tokenize_sentences("Hello world")
        assert result == ["Hello world"]

    def test_period_splits_two_sentences(self):
        result = _tokenize_sentences("First. Second.")
        assert "First" in result
        assert "Second" in result

    def test_question_mark_splits(self):
        result = _tokenize_sentences("How are you? Fine.")
        assert len(result) == 2

    def test_exclamation_splits(self):
        result = _tokenize_sentences("Watch out! Too late.")
        assert len(result) == 2

    def test_consecutive_terminators_treated_as_one(self):
        # "!?" should produce the same number of splits as "!"
        result = _tokenize_sentences("Stop!? Never.")
        assert len(result) == 2

    def test_strips_whitespace_from_segments(self):
        for segment in _tokenize_sentences("  First.  Second.  "):
            assert segment == segment.strip()

    def test_whitespace_only_segments_not_included(self):
        # A trailing period means the last split is empty — should be excluded
        result = _tokenize_sentences("One. Two.")
        assert all(s.strip() for s in result)

    def test_only_terminators(self):
        # A string of just punctuation produces no non-empty segments
        assert _tokenize_sentences("...") == []

    def test_multiline_text(self):
        result = _tokenize_sentences("Line one.\nLine two.")
        assert len(result) == 2


# ── _tokenize_words ───────────────────────────────────────────────────────────


class TestTokenizeWords:
    def test_empty_string_returns_empty_list(self):
        assert _tokenize_words("") == []

    def test_basic_words_lowercased(self):
        result = _tokenize_words("Hello World")
        assert result == ["hello", "world"]

    def test_punctuation_stripped(self):
        result = _tokenize_words("Hello, world!")
        assert result == ["hello", "world"]

    def test_numbers_included_as_words(self):
        # \w matches digits too, so "42" is a word token
        result = _tokenize_words("Chapter 42")
        assert "42" in result

    def test_hyphenated_word_splits(self):
        # \b\w+\b won't cross a hyphen — each part is its own token
        result = _tokenize_words("well-known")
        assert "well" in result
        assert "known" in result

    def test_all_punctuation_returns_empty(self):
        assert _tokenize_words("...!!!???") == []

    def test_preserves_word_order(self):
        result = _tokenize_words("the quick brown fox")
        assert result == ["the", "quick", "brown", "fox"]


# ── _ngrams ───────────────────────────────────────────────────────────────────


class TestNgrams:
    def test_empty_words_returns_empty(self):
        assert _ngrams([], 2) == []

    def test_fewer_words_than_n_returns_empty(self):
        assert _ngrams(["only"], 2) == []
        assert _ngrams(["a", "b"], 3) == []

    def test_exactly_n_words_returns_one_gram(self):
        result = _ngrams(["a", "b"], 2)
        assert result == [("a", "b")]

    def test_bigrams_count(self):
        words = ["a", "b", "c", "d"]
        result = _ngrams(words, 2)
        assert len(result) == 3
        assert ("a", "b") in result
        assert ("c", "d") in result

    def test_trigrams_count(self):
        words = ["a", "b", "c", "d"]
        result = _ngrams(words, 3)
        assert len(result) == 2
        assert ("a", "b", "c") in result
        assert ("b", "c", "d") in result

    def test_unigrams(self):
        words = ["x", "y"]
        result = _ngrams(words, 1)
        assert result == [("x",), ("y",)]

    def test_sliding_window_is_correct(self):
        words = ["w1", "w2", "w3"]
        result = _ngrams(words, 2)
        assert result[0] == ("w1", "w2")
        assert result[1] == ("w2", "w3")


# ── _cosine_similarity ────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_both_empty_counters_return_zero(self):
        assert _cosine_similarity(Counter(), Counter()) == 0.0

    def test_first_empty_returns_zero(self):
        assert _cosine_similarity(Counter(), Counter(a=1)) == 0.0

    def test_second_empty_returns_zero(self):
        assert _cosine_similarity(Counter(a=1), Counter()) == 0.0

    def test_identical_counters_return_one(self):
        c = Counter(a=3, b=1, c=2)
        assert _cosine_similarity(c, c) == pytest.approx(1.0)

    def test_identical_copy_returns_one(self):
        c = Counter({"x": 5, "y": 3})
        d = Counter({"x": 5, "y": 3})
        assert _cosine_similarity(c, d) == pytest.approx(1.0)

    def test_orthogonal_counters_return_zero(self):
        # No shared keys → dot product is 0
        a = Counter({"x": 10})
        b = Counter({"y": 10})
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        a = Counter({"a": 2, "b": 3, "c": 1})
        b = Counter({"a": 1, "b": 1, "d": 5})
        result = _cosine_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_scaled_counter_returns_one(self):
        # Scaling all values by a constant doesn't change direction
        a = Counter({"p": 2, "q": 4})
        b = Counter({"p": 10, "q": 20})
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_partial_overlap_between_zero_and_one(self):
        a = Counter({"a": 1, "b": 1})
        b = Counter({"a": 1, "c": 1})
        result = _cosine_similarity(a, b)
        assert 0.0 < result < 1.0

    def test_symmetry(self):
        a = Counter({"x": 3, "y": 1})
        b = Counter({"x": 1, "z": 2})
        assert _cosine_similarity(a, b) == pytest.approx(_cosine_similarity(b, a))


# ── StyleProfile.fit() ────────────────────────────────────────────────────────


class TestStyleProfileFit:
    def test_returns_self_for_chaining(self):
        p = StyleProfile()
        result = p.fit(["Some sample text."])
        assert result is p

    def test_empty_sample_list_leaves_avg_sentence_length_zero(self):
        p = StyleProfile().fit([])
        assert p.avg_sentence_length == 0.0

    def test_empty_sample_list_leaves_bigrams_empty(self):
        p = StyleProfile().fit([])
        assert p.bigram_freq == Counter()

    def test_empty_sample_list_leaves_trigrams_empty(self):
        p = StyleProfile().fit([])
        assert p.trigram_freq == Counter()

    def test_empty_sample_list_sets_tone_fingerprint(self):
        # fit([]) still computes and stores a tone_fingerprint (all-zero values)
        p = StyleProfile().fit([])
        assert isinstance(p.tone_fingerprint, dict)
        assert "positivity" in p.tone_fingerprint
        assert "avg_word_length" in p.tone_fingerprint

    def test_single_word_sample_avg_sentence_length_one(self):
        p = StyleProfile().fit(["hello"])
        assert p.avg_sentence_length == pytest.approx(1.0)

    def test_single_sentence_avg_length_correct(self):
        # "the cat sat on the mat" = 6 words
        p = StyleProfile().fit(["The cat sat on the mat."])
        assert p.avg_sentence_length == pytest.approx(6.0)

    def test_two_sentences_avg_length_averaged(self):
        # 6 words + 6 words → average 6.0
        p = StyleProfile().fit(["The cat sat on the mat. The dog lay on the rug."])
        assert p.avg_sentence_length == pytest.approx(6.0)

    def test_multiple_samples_sentences_pooled(self):
        # "The cat sat on the mat" = 6 words, "Dogs run" = 2 words → average 4.0
        # (period is punctuation, not a word token)
        p = StyleProfile().fit(["The cat sat on the mat.", "Dogs run."])
        assert p.avg_sentence_length == pytest.approx(4.0)

    def test_bigrams_computed_from_words(self):
        p = StyleProfile().fit(["the cat sat"])
        assert p.bigram_freq[("the", "cat")] == 1
        assert p.bigram_freq[("cat", "sat")] == 1

    def test_bigrams_accumulate_across_samples(self):
        # "the cat" appears once in each sample
        p = StyleProfile().fit(["the cat ran", "the cat sat"])
        assert p.bigram_freq[("the", "cat")] == 2

    def test_trigrams_computed(self):
        p = StyleProfile().fit(["alpha beta gamma delta"])
        assert p.trigram_freq[("alpha", "beta", "gamma")] == 1
        assert p.trigram_freq[("beta", "gamma", "delta")] == 1

    def test_single_word_has_no_bigrams(self):
        p = StyleProfile().fit(["word"])
        assert p.bigram_freq == Counter()

    def test_two_words_has_one_bigram_no_trigram(self):
        p = StyleProfile().fit(["hello world"])
        assert len(p.bigram_freq) == 1
        assert p.trigram_freq == Counter()

    def test_connector_words_counted(self):
        p = StyleProfile().fit(["I went; however, it was closed. Therefore, I stayed."])
        assert p.connector_freq["however"] == 1
        assert p.connector_freq["therefore"] == 1

    def test_positive_tone_fingerprint(self):
        p = StyleProfile().fit(["This is great and wonderful and excellent."])
        assert p.tone_fingerprint["positivity"] > 0.0

    def test_negative_tone_fingerprint(self):
        p = StyleProfile().fit(["This is terrible and awful and wrong."])
        assert p.tone_fingerprint["negativity"] > 0.0

    def test_question_density_from_question_marks(self):
        p = StyleProfile().fit(["Is this working? Can we test it?"])
        assert p.tone_fingerprint["question_density"] > 0.0

    def test_exclaim_density_from_exclamation_marks(self):
        p = StyleProfile().fit(["Amazing! Incredible!"])
        assert p.tone_fingerprint["exclaim_density"] > 0.0

    def test_avg_word_length_computed(self):
        # "ab" = 2, "cd" = 2, "ef" = 2 → avg = 2.0
        p = StyleProfile().fit(["ab cd ef"])
        assert p.tone_fingerprint["avg_word_length"] == pytest.approx(2.0)

    def test_fit_overwrites_previous_state(self):
        p = StyleProfile().fit(["First sample text here."])
        original_sl = p.avg_sentence_length
        p.fit(["X."])  # one-word sentence
        assert p.avg_sentence_length != original_sl
        assert p.avg_sentence_length == pytest.approx(1.0)

    def test_empty_string_sample_treated_like_no_words(self):
        p = StyleProfile().fit([""])
        assert p.avg_sentence_length == 0.0
        assert p.bigram_freq == Counter()

    def test_tone_fingerprint_has_five_keys(self):
        p = StyleProfile().fit(["Some text here."])
        expected_keys = {
            "positivity", "negativity", "question_density",
            "exclaim_density", "avg_word_length",
        }
        assert set(p.tone_fingerprint.keys()) == expected_keys

    def test_all_tone_fingerprint_values_are_floats(self):
        p = StyleProfile().fit(["The quick brown fox."])
        for v in p.tone_fingerprint.values():
            assert isinstance(v, float)

    def test_positivity_and_negativity_sum_at_most_one(self):
        p = StyleProfile().fit(["good bad great terrible excellent awful"])
        assert p.tone_fingerprint["positivity"] + p.tone_fingerprint["negativity"] <= 1.0


# ── StyleProfile.similarity_score() ──────────────────────────────────────────


class TestSimilarityScore:
    def test_unfitted_profile_returns_zero(self):
        p = StyleProfile()  # default: tone_fingerprint = {}
        assert p.similarity_score("any text at all") == 0.0

    def test_same_text_scores_one(self):
        # Need enough words for bigrams to match perfectly
        text = "The cat sat on the mat. The dog lay on the rug."
        p = StyleProfile().fit([text])
        assert p.similarity_score(text) == pytest.approx(1.0)

    def test_score_in_unit_interval_normal_text(self):
        p = StyleProfile().fit(["The quick brown fox jumps over the lazy dog."])
        score = p.similarity_score("The slow red cat walks under the fence.")
        assert 0.0 <= score <= 1.0

    def test_score_in_unit_interval_empty_input(self):
        p = StyleProfile().fit(["Some sample text."])
        score = p.similarity_score("")
        assert 0.0 <= score <= 1.0

    def test_score_in_unit_interval_gibberish(self):
        p = StyleProfile().fit(["Normal everyday sentences."])
        score = p.similarity_score("xyzzy qwerty asdfgh jklzxcvbn")
        assert 0.0 <= score <= 1.0

    def test_different_text_scores_less_than_same_text(self):
        training = (
            "The cat sat on the mat. The dog lay on the rug. "
            "The bird sang in the tree. The fish swam in the pond."
        )
        p = StyleProfile().fit([training])
        same_score = p.similarity_score(training)
        # Completely different vocabulary, length, and style
        different = (
            "Quantum mechanics fundamentally challenges our understanding of reality! "
            "Wave-particle duality is extraordinary? Truly remarkable phenomena."
        )
        different_score = p.similarity_score(different)
        assert same_score > different_score

    def test_score_does_not_modify_profile(self):
        p = StyleProfile().fit(["The quick brown fox."])
        sl_before = p.avg_sentence_length
        fp_before = dict(p.tone_fingerprint)
        bigrams_before = dict(p.bigram_freq)
        p.similarity_score("Some unrelated text that has nothing to do with training.")
        assert p.avg_sentence_length == sl_before
        assert p.tone_fingerprint == fp_before
        assert p.bigram_freq == bigrams_before

    def test_empty_string_after_fit_does_not_raise(self):
        p = StyleProfile().fit(["Normal text here."])
        score = p.similarity_score("")
        assert isinstance(score, float)

    def test_single_word_input_does_not_raise(self):
        p = StyleProfile().fit(["Normal everyday text with multiple words."])
        score = p.similarity_score("hello")
        assert 0.0 <= score <= 1.0

    def test_score_after_fit_empty_list_does_not_raise(self):
        # fit([]) sets tone_fingerprint to non-empty dict, so score runs without error
        p = StyleProfile().fit([])
        score = p.similarity_score("Any text")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_only_punctuation_input(self):
        p = StyleProfile().fit(["The cat sat on the mat."])
        score = p.similarity_score("...!!!???")
        assert 0.0 <= score <= 1.0

    def test_very_long_text_does_not_raise(self):
        sample = "The quick brown fox jumps over the lazy dog. " * 50
        p = StyleProfile().fit([sample])
        score = p.similarity_score(sample * 2)
        assert 0.0 <= score <= 1.0


# ── Monotonicity and ordering guarantees ──────────────────────────────────────


class TestMonotonicity:
    """Verify the core similarity ordering property."""

    _TRAINING = (
        "The elderly professor sat quietly at his oak desk reviewing the students' papers. "
        "He carefully noted the errors with his red pen. "
        "The afternoon light faded slowly through the dusty window panes. "
        "His coffee had long since grown cold beside the lamp."
    )

    def _profile(self) -> StyleProfile:
        return StyleProfile().fit([self._TRAINING])

    def test_identical_text_scores_highest(self):
        p = self._profile()
        # Same text must score 1.0 — if not, the implementation is broken
        assert p.similarity_score(self._TRAINING) == pytest.approx(1.0)

    def test_vastly_different_style_scores_below_identical(self):
        p = self._profile()
        same = p.similarity_score(self._TRAINING)
        # Capslock-heavy, exclamation-heavy, totally different vocabulary
        alien = (
            "QUANTUM PHYSICS IS AMAZING!!! Particle accelerators smash protons! "
            "Incredible electromagnetic phenomena occur! Science is extraordinary!!!"
        )
        assert same > p.similarity_score(alien)

    def test_similar_style_scores_above_opposite_style(self):
        p = self._profile()
        # Stylistically similar: calm, past-tense narrative, similar sentence length
        similar = (
            "The young librarian sorted the books on the wooden shelf carefully. "
            "She marked the worn volumes with small sticky notes. "
            "The reading room was quiet in the pale morning light."
        )
        # Stylistically opposite: questions, exclamations, short punchy sentences
        opposite = "Really?! Why?! Stop! Go! Now! Please!"
        assert p.similarity_score(similar) > p.similarity_score(opposite)

    def test_score_increases_with_vocabulary_overlap(self):
        """Re-using words from the training set should raise the bigram/trigram score."""
        p = self._profile()
        # Extract actual words from training and build a new sentence
        overlapping = (
            "The professor sat at his desk reviewing papers carefully with his pen."
        )
        # Completely foreign vocabulary
        non_overlapping = "Nebulae collapse forming stellar nurseries in galactic spirals."
        assert p.similarity_score(overlapping) > p.similarity_score(non_overlapping)

    def test_matching_sentence_length_helps_score(self):
        """A text whose sentence length matches the training average should score better
        on the sentence-length component than one that wildly differs."""
        # Training has ~10-word sentences on average; score a 1-word vs ~10-word text
        p = self._profile()
        score_long = p.similarity_score(
            "The professor reviewed papers carefully at his old wooden desk."
        )
        score_one_word = p.similarity_score("Yes.")
        assert score_long > score_one_word


# ── Single-word corpus edge cases ─────────────────────────────────────────────


class TestSingleWordCorpus:
    """Corpus with only one word has no bigrams or trigrams.
    Verify the profile still behaves sensibly."""

    def test_fit_single_word_no_bigrams(self):
        p = StyleProfile().fit(["hello"])
        assert p.bigram_freq == Counter()

    def test_fit_single_word_no_trigrams(self):
        p = StyleProfile().fit(["hello"])
        assert p.trigram_freq == Counter()

    def test_same_word_similarity_below_one(self):
        # bigram + trigram components are zero, so max is 0.30+0.30 = 0.60
        p = StyleProfile().fit(["good"])
        score = p.similarity_score("good")
        # Must be >= 0.50 (sl and tone match) and < 1.0 (bigrams don't)
        assert 0.50 <= score < 1.0

    def test_different_word_scores_lower_than_same_word(self):
        p = StyleProfile().fit(["good"])
        same = p.similarity_score("good")
        # "terrible" has: different tone (negative), different word length
        diff = p.similarity_score("terrible")
        assert same > diff

    def test_score_in_unit_interval(self):
        p = StyleProfile().fit(["hello"])
        assert 0.0 <= p.similarity_score("world") <= 1.0


# ── Tone fingerprint boundary cases ───────────────────────────────────────────


class TestToneFingerprint:
    def test_positive_heavy_text_has_high_positivity(self):
        p = StyleProfile().fit(["good great excellent wonderful amazing beautiful"])
        assert p.tone_fingerprint["positivity"] == pytest.approx(1.0)

    def test_negative_heavy_text_has_high_negativity(self):
        p = StyleProfile().fit(["bad terrible awful horrible wrong failure"])
        assert p.tone_fingerprint["negativity"] == pytest.approx(1.0)

    def test_question_heavy_text_has_nonzero_question_density(self):
        p = StyleProfile().fit(["Is this? Are you? Can we?"])
        assert p.tone_fingerprint["question_density"] > 0.0

    def test_no_questions_gives_zero_question_density(self):
        p = StyleProfile().fit(["The cat sat on the mat."])
        assert p.tone_fingerprint["question_density"] == 0.0

    def test_exclaim_heavy_text_has_nonzero_exclaim_density(self):
        p = StyleProfile().fit(["Watch out! Go now! Run!"])
        assert p.tone_fingerprint["exclaim_density"] > 0.0

    def test_longer_words_raise_avg_word_length(self):
        p_short = StyleProfile().fit(["a b c d e"])
        p_long = StyleProfile().fit(["extraordinary magnificent spectacular"])
        assert (
            p_long.tone_fingerprint["avg_word_length"]
            > p_short.tone_fingerprint["avg_word_length"]
        )

    def test_similarity_identical_positive_sentiment(self):
        text = "This is good and great and excellent."
        p = StyleProfile().fit([text])
        assert p.similarity_score(text) == pytest.approx(1.0)

    def test_positive_profile_penalises_negative_text(self):
        positive = "good great excellent wonderful amazing"
        negative = "bad terrible awful horrible failure"
        p = StyleProfile().fit([positive])
        score_pos = p.similarity_score(positive)
        score_neg = p.similarity_score(negative)
        assert score_pos > score_neg


# ── Score clamping ────────────────────────────────────────────────────────────


class TestScoreClamping:
    """similarity_score must always return a value in [0.0, 1.0]."""

    CASES = [
        "",
        "x",
        "a b c",
        "The quick brown fox.",
        "!!??...",
        "good great excellent wonderful",
        "bad terrible awful horrible",
        "Is this right? Maybe not!",
        " ".join(["word"] * 200),
    ]

    @pytest.mark.parametrize("text", CASES)
    def test_score_clamped(self, text: str):
        p = StyleProfile().fit(["The cat sat on the mat. The dog ran down the lane."])
        score = p.similarity_score(text)
        assert 0.0 <= score <= 1.0, f"Out of range for input {text!r}: {score}"
