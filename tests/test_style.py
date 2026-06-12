"""Adversarial test suite for writing_assistant.style.StyleProfile.

Covers:
  - learn() produces non-default, correct marker values for controlled samples
  - JSON round-trip preserves every marker value exactly
  - StyleProfile attached to Pipeline injects profile.summary() into the
    consistency-pass prompt (verified via a capturing MockLLM subclass)
  - Corrupting individual markers changes the injected prompt, proving the
    markers are actually used rather than silently ignored
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from writing_assistant.style import StyleProfile
from mock_llm import MockLLM
from writing_assistant.passes import CLARITY, CONSISTENCY
from writing_assistant.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class CapturingMockLLM(MockLLM):
    """MockLLM subclass that records every prompt it receives."""

    def __init__(self, response: str = "unchanged text") -> None:
        super().__init__(responses=[response])
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt)


def _run_consistency(profile: StyleProfile, text: str = "Some text.") -> str:
    """Run a single-pass consistency pipeline and return the captured prompt."""
    llm = CapturingMockLLM()
    Pipeline(passes=[CONSISTENCY], backend=llm, style_profile=profile).run(text)
    assert llm.prompts, "generate() was never called"
    return llm.prompts[0]


# ---------------------------------------------------------------------------
# Controlled sample corpus — hand-counted for deterministic expectations
# ---------------------------------------------------------------------------

# Three sentences, all active voice:
#   "Dogs run fast."              → 3 words, no passive
#   "However, cats sleep often."  → 4 words, no passive;  transition: however
#   "Therefore, birds fly high."  → 4 words, no passive;  transition: therefore
#
# avg_sentence_length  = (3+4+4) / 3 = 11/3 ≈ 3.667
# passive_voice_ratio  = 0 / 3 = 0.0
# all words: dogs run fast however cats sleep often therefore birds fly high
#   = 11 tokens, 11 unique → vocab_richness = 1.0
# preferred_transition_words = ["however", "therefore"]
ACTIVE_SAMPLE = ["Dogs run fast. However, cats sleep often. Therefore, birds fly high."]

# Two fully passive sentences (past participles ending in -ed):
#   "The report was analyzed by Alice."  → 6 words, passive: "was analyzed"
#   "Furthermore, errors were detected." → 4 words, passive: "were detected"
#
# avg_sentence_length  = (6+4) / 2 = 5.0
# passive_voice_ratio  = 2 / 2 = 1.0
# all words: the report was analyzed by alice furthermore errors were detected
#   = 10 tokens, 10 unique → vocab_richness = 1.0
# preferred_transition_words = ["furthermore"]
PASSIVE_SAMPLE = [
    "The report was analyzed by Alice. Furthermore, errors were detected."
]


# ---------------------------------------------------------------------------
# learn() — non-default values
# ---------------------------------------------------------------------------


class TestLearnProducesNonDefaultValues:
    """learn() must compute real metrics, not return the default zeros."""

    def test_avg_sentence_length_nonzero(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.avg_sentence_length > 0.0

    def test_vocabulary_richness_nonzero(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.vocabulary_richness > 0.0

    def test_preferred_transition_words_populated(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert len(profile.preferred_transition_words) > 0

    def test_passive_ratio_nonzero_for_passive_text(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert profile.passive_voice_ratio > 0.0

    def test_passive_ratio_is_one_for_all_passive_sentences(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert profile.passive_voice_ratio == pytest.approx(1.0)

    def test_passive_ratio_zero_for_active_text(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.passive_voice_ratio == pytest.approx(0.0)

    def test_result_is_styleprofile_instance(self) -> None:
        assert isinstance(StyleProfile.learn(ACTIVE_SAMPLE), StyleProfile)


# ---------------------------------------------------------------------------
# learn() — exact values for the controlled corpus
# ---------------------------------------------------------------------------


class TestLearnExactValues:
    """Computed values must match hand-calculated expectations."""

    def test_avg_sentence_length_active_sample(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.avg_sentence_length == pytest.approx(11 / 3, rel=1e-6)

    def test_avg_sentence_length_passive_sample(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert profile.avg_sentence_length == pytest.approx(5.0, rel=1e-6)

    def test_vocab_richness_all_unique_words(self) -> None:
        # Every word in ACTIVE_SAMPLE is unique → richness = 1.0
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.vocabulary_richness == pytest.approx(1.0)

    def test_transition_words_active_sample(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert profile.preferred_transition_words == ["however", "therefore"]

    def test_transition_words_passive_sample(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert profile.preferred_transition_words == ["furthermore"]

    def test_transition_words_are_sorted(self) -> None:
        sample = ["However, this. Therefore, we conclude. Moreover, that is so."]
        profile = StyleProfile.learn(sample)
        tw = profile.preferred_transition_words
        assert tw == sorted(tw), "transition words must be lexicographically sorted"

    def test_passive_ratio_one_for_passive_sample(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert profile.passive_voice_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# learn() — edge cases
# ---------------------------------------------------------------------------


class TestLearnEdgeCases:
    """learn() must handle boundary inputs without crashing or producing garbage."""

    def test_empty_list_returns_defaults(self) -> None:
        profile = StyleProfile.learn([])
        assert profile.avg_sentence_length == 0.0
        assert profile.passive_voice_ratio == 0.0
        assert profile.vocabulary_richness == 0.0
        assert profile.preferred_transition_words == []

    def test_empty_string_returns_defaults(self) -> None:
        profile = StyleProfile.learn([""])
        assert profile.avg_sentence_length == 0.0
        assert profile.passive_voice_ratio == 0.0
        assert profile.vocabulary_richness == 0.0
        assert profile.preferred_transition_words == []

    def test_whitespace_only_string_returns_defaults(self) -> None:
        profile = StyleProfile.learn(["   \t\n  "])
        assert profile.avg_sentence_length == 0.0
        assert profile.vocabulary_richness == 0.0

    def test_single_word_sentence(self) -> None:
        profile = StyleProfile.learn(["Go."])
        assert profile.avg_sentence_length == pytest.approx(1.0)
        assert profile.vocabulary_richness == pytest.approx(1.0)

    def test_repeated_word_reduces_richness(self) -> None:
        # 6 tokens, 1 unique → richness = 1/6
        profile = StyleProfile.learn(["the the the the the the."])
        assert profile.vocabulary_richness == pytest.approx(1 / 6, rel=1e-6)

    def test_no_transition_words_in_plain_text(self) -> None:
        profile = StyleProfile.learn(["Dogs run fast. Cats sleep here."])
        assert profile.preferred_transition_words == []

    def test_multiple_samples_are_joined(self) -> None:
        # "however" only appears in the second sample
        joint = StyleProfile.learn(["Dogs run fast.", "However, cats sleep often."])
        first_only = StyleProfile.learn(["Dogs run fast."])
        assert "however" in joint.preferred_transition_words
        assert "however" not in first_only.preferred_transition_words

    def test_transition_word_not_duplicated_when_repeated(self) -> None:
        profile = StyleProfile.learn(["However, this. However, that."])
        assert profile.preferred_transition_words.count("however") == 1

    def test_case_insensitive_transition_detection(self) -> None:
        profile = StyleProfile.learn(["HOWEVER, dogs run. THEREFORE, cats sleep."])
        assert "however" in profile.preferred_transition_words
        assert "therefore" in profile.preferred_transition_words

    def test_transition_words_stored_lowercase(self) -> None:
        profile = StyleProfile.learn(["HOWEVER, yes. MOREOVER, no."])
        for word in profile.preferred_transition_words:
            assert word == word.lower(), f"Expected lowercase, got {word!r}"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """to_json() / from_json() must preserve every marker value exactly."""

    def test_roundtrip_preserves_avg_sentence_length(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        restored = StyleProfile.from_json(profile.to_json())
        assert restored.avg_sentence_length == profile.avg_sentence_length

    def test_roundtrip_preserves_passive_voice_ratio(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        restored = StyleProfile.from_json(profile.to_json())
        assert restored.passive_voice_ratio == profile.passive_voice_ratio

    def test_roundtrip_preserves_vocabulary_richness(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        restored = StyleProfile.from_json(profile.to_json())
        assert restored.vocabulary_richness == profile.vocabulary_richness

    def test_roundtrip_preserves_transition_words_list(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        assert (
            StyleProfile.from_json(profile.to_json()).preferred_transition_words
            == profile.preferred_transition_words
        )

    def test_roundtrip_full_equality(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        assert StyleProfile.from_json(profile.to_json()) == profile

    def test_roundtrip_default_profile(self) -> None:
        profile = StyleProfile()
        assert StyleProfile.from_json(profile.to_json()) == profile

    def test_roundtrip_manually_constructed_profile(self) -> None:
        profile = StyleProfile(
            avg_sentence_length=12.345678,
            passive_voice_ratio=0.333333,
            vocabulary_richness=0.666666,
            preferred_transition_words=["thus", "moreover"],
        )
        restored = StyleProfile.from_json(profile.to_json())
        assert restored.avg_sentence_length == profile.avg_sentence_length
        assert restored.passive_voice_ratio == profile.passive_voice_ratio
        assert restored.vocabulary_richness == profile.vocabulary_richness
        assert restored.preferred_transition_words == profile.preferred_transition_words

    def test_to_json_produces_valid_json(self) -> None:
        raw = StyleProfile.learn(ACTIVE_SAMPLE).to_json()
        parsed = json.loads(raw)  # must not raise
        assert set(parsed.keys()) == {
            "avg_sentence_length",
            "passive_voice_ratio",
            "vocabulary_richness",
            "preferred_transition_words",
        }

    def test_roundtrip_preserves_empty_transition_list(self) -> None:
        profile = StyleProfile(preferred_transition_words=[])
        assert StyleProfile.from_json(profile.to_json()).preferred_transition_words == []

    def test_roundtrip_preserves_transition_word_order(self) -> None:
        # Order must survive even if it differs from alphabetical
        profile = StyleProfile(preferred_transition_words=["thus", "also", "moreover"])
        assert (
            StyleProfile.from_json(profile.to_json()).preferred_transition_words
            == ["thus", "also", "moreover"]
        )

    def test_from_json_rejects_invalid_json(self) -> None:
        with pytest.raises(Exception):
            StyleProfile.from_json("not-json{{{")


# ---------------------------------------------------------------------------
# Pipeline: consistency pass injects profile summary
# ---------------------------------------------------------------------------


class TestPipelineConsistencyPassInjectsProfile:
    """Consistency-pass prompt must embed profile.summary() when a profile is attached."""

    def test_prompt_contains_profile_summary_string(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert profile.summary() in prompt

    def test_prompt_contains_style_profile_header(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert "Style profile:" in prompt

    def test_prompt_contains_avg_sentence_length_line(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert "Average sentence length:" in prompt

    def test_prompt_contains_passive_voice_ratio_line(self) -> None:
        profile = StyleProfile.learn(PASSIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert "Passive voice ratio:" in prompt

    def test_prompt_contains_vocabulary_richness_line(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert "Vocabulary richness" in prompt

    def test_prompt_contains_specific_transition_words(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        assert "however" in prompt
        assert "therefore" in prompt

    def test_profile_summary_appears_before_instructions(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        prompt = _run_consistency(profile)
        summary_pos = prompt.index(profile.summary())
        instructions_pos = prompt.index("Rewrite the following text")
        assert summary_pos < instructions_pos, (
            "Style profile must be injected before the pass instructions"
        )

    def test_no_profile_means_no_style_section(self) -> None:
        llm = CapturingMockLLM()
        Pipeline(passes=[CONSISTENCY], backend=llm, style_profile=None).run("Text.")
        assert "Style profile:" not in llm.prompts[0]

    def test_non_consistency_pass_does_not_inject_profile(self) -> None:
        profile = StyleProfile.learn(ACTIVE_SAMPLE)
        llm = CapturingMockLLM()
        Pipeline(passes=[CLARITY], backend=llm, style_profile=profile).run("Text.")
        assert "Style profile:" not in llm.prompts[0]
        assert profile.summary() not in llm.prompts[0]

    def test_profile_not_injected_for_non_consistency_pass_even_with_profile(self) -> None:
        profile = StyleProfile(avg_sentence_length=42.0)
        llm = CapturingMockLLM()
        Pipeline(passes=[CLARITY], backend=llm, style_profile=profile).run("Text.")
        assert "42.0" not in llm.prompts[0]

    def test_generate_called_exactly_once_per_pass(self) -> None:
        llm = CapturingMockLLM()
        Pipeline(
            passes=[CONSISTENCY],
            backend=llm,
            style_profile=StyleProfile.learn(ACTIVE_SAMPLE),
        ).run("Text.")
        assert len(llm.prompts) == 1


# ---------------------------------------------------------------------------
# Corrupting markers changes the injected prompt (mutation resistance)
# ---------------------------------------------------------------------------


class TestCorruptedMarkerChangesPrompt:
    """Each marker must appear in the injected prompt so mutations are detectable."""

    def test_zeroing_avg_sentence_length_changes_prompt(self) -> None:
        base = StyleProfile.learn(ACTIVE_SAMPLE)
        corrupted = replace(base, avg_sentence_length=0.0)
        assert _run_consistency(base) != _run_consistency(corrupted)

    def test_zeroing_passive_ratio_changes_prompt(self) -> None:
        base = StyleProfile.learn(PASSIVE_SAMPLE)
        corrupted = replace(base, passive_voice_ratio=0.0)
        assert _run_consistency(base) != _run_consistency(corrupted)

    def test_zeroing_vocab_richness_changes_prompt(self) -> None:
        base = StyleProfile.learn(ACTIVE_SAMPLE)
        corrupted = replace(base, vocabulary_richness=0.0)
        assert _run_consistency(base) != _run_consistency(corrupted)

    def test_removing_transition_words_changes_prompt(self) -> None:
        base = StyleProfile.learn(ACTIVE_SAMPLE)
        corrupted = replace(base, preferred_transition_words=[])
        assert _run_consistency(base) != _run_consistency(corrupted)

    def test_replacing_transition_word_changes_prompt(self) -> None:
        base = StyleProfile(preferred_transition_words=["however"])
        swapped = replace(base, preferred_transition_words=["thus"])
        assert _run_consistency(base) != _run_consistency(swapped)

    def test_extreme_avg_sentence_length_visible_in_prompt(self) -> None:
        profile = StyleProfile(avg_sentence_length=999.9)
        assert "999.9" in _run_consistency(profile)

    def test_full_passive_ratio_shows_100_percent(self) -> None:
        profile = StyleProfile(passive_voice_ratio=1.0)
        assert "100.0%" in _run_consistency(profile)

    def test_zero_passive_ratio_differs_from_nonzero(self) -> None:
        zero = StyleProfile(passive_voice_ratio=0.0)
        nonzero = StyleProfile(passive_voice_ratio=0.5)
        assert _run_consistency(zero) != _run_consistency(nonzero)

    def test_adding_transition_word_changes_prompt(self) -> None:
        base = StyleProfile(preferred_transition_words=["however"])
        extended = replace(base, preferred_transition_words=["however", "therefore"])
        assert _run_consistency(base) != _run_consistency(extended)

    def test_all_four_markers_independently_affect_prompt(self) -> None:
        """Each marker individually changes the prompt — no marker is inert."""
        base = StyleProfile(
            avg_sentence_length=5.0,
            passive_voice_ratio=0.2,
            vocabulary_richness=0.6,
            preferred_transition_words=["however"],
        )
        base_prompt = _run_consistency(base)
        mutations = [
            replace(base, avg_sentence_length=0.0),
            replace(base, passive_voice_ratio=0.0),
            replace(base, vocabulary_richness=0.0),
            replace(base, preferred_transition_words=[]),
        ]
        for mutant in mutations:
            assert _run_consistency(mutant) != base_prompt


# ---------------------------------------------------------------------------
# summary() format contract
# ---------------------------------------------------------------------------


class TestSummaryFormat:
    """summary() must produce well-formed, human-readable output."""

    def test_contains_avg_sentence_length_value(self) -> None:
        profile = StyleProfile(avg_sentence_length=7.5)
        assert "7.5" in profile.summary()

    def test_passive_voice_shown_as_percentage(self) -> None:
        profile = StyleProfile(passive_voice_ratio=0.25)
        assert "25.0%" in profile.summary()

    def test_vocab_richness_two_decimal_places(self) -> None:
        # 0.4267 rounds to 0.43
        profile = StyleProfile(vocabulary_richness=0.4267)
        assert "0.43" in profile.summary()

    def test_no_transition_section_when_list_empty(self) -> None:
        profile = StyleProfile(preferred_transition_words=[])
        assert "transition" not in profile.summary().lower()

    def test_transition_words_appear_when_present(self) -> None:
        profile = StyleProfile(preferred_transition_words=["however", "thus"])
        summary = profile.summary()
        assert "however" in summary
        assert "thus" in summary

    def test_summary_is_multiline(self) -> None:
        profile = StyleProfile(avg_sentence_length=5.0, vocabulary_richness=0.5)
        assert "\n" in profile.summary()

    def test_summary_differs_across_avg_sentence_length(self) -> None:
        a = StyleProfile(avg_sentence_length=5.0)
        b = StyleProfile(avg_sentence_length=10.0)
        assert a.summary() != b.summary()

    def test_summary_differs_across_passive_ratio(self) -> None:
        a = StyleProfile(passive_voice_ratio=0.0)
        b = StyleProfile(passive_voice_ratio=0.5)
        assert a.summary() != b.summary()

    def test_zero_passive_shows_zero_percent(self) -> None:
        profile = StyleProfile(passive_voice_ratio=0.0)
        assert "0%" in profile.summary()
