"""Behavioural tests for the deterministic rule-based backend.

The backend is real shipped code (it powers the offline CLI mode and the
acceptance demo), so every rule family is pinned by an observable
transformation — not by internals.
"""
from __future__ import annotations

from writing_assistant.llm.rule_based import RuleBasedRewriter
from writing_assistant.passes import ADVERSARIAL, CLARITY, CONCISENESS, CONSISTENCY, TONE
from writing_assistant.pipeline import Pipeline
from writing_assistant.types import LLMBackend


def _prompt(pass_obj, text: str) -> str:
    """Build the exact prompt shape the pipeline sends for a normal pass."""
    return (
        f"Rewrite pass: {pass_obj.name}\nPayload characters: {len(text)}\n\n"
        f"Pass instructions:\n{pass_obj.instructions}\n\nPayload:\n{text}"
    )


def _rewrite(pass_obj, text: str) -> str:
    return RuleBasedRewriter().generate(_prompt(pass_obj, text))


class TestProtocol:
    def test_satisfies_llm_backend_protocol(self) -> None:
        backend: LLMBackend = RuleBasedRewriter()
        assert isinstance(backend.generate("Text:\nx"), str)


class TestClarityRules:
    def test_in_the_event_that_becomes_if(self) -> None:
        out = _rewrite(CLARITY, "In the event that it rains, stay home.")
        assert out == "If it rains, stay home."

    def test_utilization_becomes_use(self) -> None:
        out = _rewrite(CLARITY, "The utilization of tools matters.")
        assert "utilization" not in out.lower()
        assert "use" in out.lower()

    def test_in_order_to_becomes_to(self) -> None:
        out = _rewrite(CLARITY, "We test in order to learn.")
        assert out == "We test to learn."

    def test_capital_preserved_on_replacement(self) -> None:
        out = _rewrite(CLARITY, "Numerous people agree.")
        assert out.startswith("Many ")

    def test_word_boundary_respected(self) -> None:
        # "commencement" must NOT be rewritten via the "commence" rule.
        out = _rewrite(CLARITY, "The commencement ceremony was long.")
        assert "commencement" in out

    def test_plural_verb_forms_covered(self) -> None:
        out = _rewrite(CLARITY, "She utilizes the tool and facilitates work.")
        assert "utilizes" not in out
        assert "uses" in out

    def test_clean_text_passes_through_unchanged(self) -> None:
        clean = "We test to learn. Plain words win."
        assert _rewrite(CLARITY, clean) == clean

    def test_declared_identity_wins_over_all_marker_collisions(self) -> None:
        collisions = (
            "adversarial editor",
            "improve clarity",
            "respectful tone",
            "more concise",
            "consistent terminology",
        )
        for collision in collisions:
            payload = "It is really useful."
            prompt = (
                f"Rewrite pass: clarity\nPayload characters: {len(payload)}\n\n"
                f"Desired style:\nPreferred vocabulary: {collision}\n\n"
                f"Pass instructions:\n{CLARITY.instructions}\n\n"
                f"Payload:\n{payload}"
            )
            assert RuleBasedRewriter().generate(prompt) == "It is really useful."


class TestConcisenessRules:
    def test_filler_clause_removed_and_sentence_recapitalised(self) -> None:
        out = _rewrite(CONCISENESS, "It should be noted that the cache is warm.")
        assert out == "The cache is warm."

    def test_each_and_every_collapsed(self) -> None:
        out = _rewrite(CONCISENESS, "Check each and every item.")
        assert out == "Check every item."

    def test_very_removed_without_breaking_every(self) -> None:
        out = _rewrite(CONCISENESS, "Every step is very important.")
        assert out == "Every step is important."

    def test_include_but_not_limited_to_collapsed(self) -> None:
        out = _rewrite(CONCISENESS, "Features include, but are not limited to, search.")
        assert out == "Features include search."

    def test_output_never_longer_than_input(self) -> None:
        wordy = "It is worth noting that basically each and every very small step really helps."
        out = _rewrite(CONCISENESS, wordy)
        assert len(out.split()) < len(wordy.split())


class TestToneRules:
    def test_contractions_expanded(self) -> None:
        out = _rewrite(TONE, "We can't ship this; it's broken.")
        assert "can't" not in out
        assert "cannot" in out
        assert "it is" in out

    def test_exclamations_calmed(self) -> None:
        out = _rewrite(TONE, "Ship it now!")
        assert "!" not in out
        assert out.endswith(".")

    def test_informal_register_lifted(self) -> None:
        out = _rewrite(TONE, "We found a lot of awesome stuff.")
        assert "a lot of" not in out
        assert "awesome" not in out


class TestConsistencyRules:
    def test_curly_quotes_normalised(self) -> None:
        out = _rewrite(CONSISTENCY, "She said “hello” and left.")
        assert '"hello"' in out

    def test_double_hyphen_becomes_em_dash(self) -> None:
        out = _rewrite(CONSISTENCY, "One thing -- the only thing.")
        assert "--" not in out
        assert "—" in out

    def test_terminology_unified(self) -> None:
        out = _rewrite(CONSISTENCY, "Send an e-mail about the web site.")
        assert "email" in out
        assert "website" in out

    def test_double_spaces_collapsed(self) -> None:
        out = _rewrite(CONSISTENCY, "One  two   three.")
        assert "  " not in out


class TestAdversarialPass:
    def test_adversarial_prompt_shape_is_handled(self) -> None:
        """The adversarial pass uses the Original/Accumulated/Current prompt
        shape; the backend must edit the CURRENT text, not the history."""
        payload = "If it rains, we basically can't go!"
        prompt = (
            f"Rewrite pass: adversarial\nPayload characters: {len(payload)}\n\n"
            "Pass instructions:\n"
            f"{ADVERSARIAL.instructions}\n\n"
            "Original text:\nIn the event that it rains.\n\n"
            "Accumulated rewrites:\n[clarity]\nIf it rains.\n\n"
            f"Payload:\n{payload}"
        )
        out = RuleBasedRewriter().generate(prompt)
        assert "basically" not in out
        assert "can't" not in out
        assert "!" not in out

    def test_adversarial_leaves_clean_text_alone(self) -> None:
        payload = "Clean prose stays clean."
        prompt = (
            f"Rewrite pass: adversarial\nPayload characters: {len(payload)}\n\n"
            "Pass instructions:\n"
            f"{ADVERSARIAL.instructions}\n\n"
            "Original text:\nMessy.\n\nAccumulated rewrites:\n[clarity]\nClean.\n\n"
            f"Payload:\n{payload}"
        )
        assert RuleBasedRewriter().generate(prompt) == "Clean prose stays clean."


class TestDeterminismAndConvergence:
    def test_same_input_same_output(self) -> None:
        text = "In the event that you utilize this, don't panic!"
        a = _rewrite(CLARITY, text)
        b = _rewrite(CLARITY, text)
        assert a == b

    def test_rules_are_idempotent(self) -> None:
        text = "In the event that it should be noted that we can't utilize very big words."
        once = _rewrite(CLARITY, text)
        twice = _rewrite(CLARITY, once)
        assert once == twice


class TestFullPipelineIntegration:
    def test_five_pass_pipeline_tightens_wordy_draft(self) -> None:
        draft = (
            "In the event that you are considering making utilization of this, "
            "it should be noted that each and every very small step really helps! "
            "Don't forget to send an e-mail."
        )
        results = Pipeline(
            passes=[CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL],
            backend=RuleBasedRewriter(),
        ).run(draft)
        final = results[-1].revised
        assert len(final.split()) < len(draft.split())
        assert "in the event that" not in final.lower()
        assert "each and every" not in final.lower()
        assert "don't" not in final.lower()
        assert "e-mail" not in final.lower()
        assert "!" not in final
