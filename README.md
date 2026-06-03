# Writing Assistant

A lightweight, multi-pass text rewriting library with a pluggable backend and style analysis.

## Install

```bash
pip install -e .
```

Python 3.10+ is required. No external dependencies beyond the standard library.

## Quick start

```python
from writing_assistant.backends import MockBackend, ClaudeCLIBackend
from writing_assistant.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(passes=["clarity", "conciseness", "adversarial"])
backend = MockBackend(responses=["Clearer text.", "Concise text.", "Final text."])
pipeline = Pipeline(config=config, backend=backend)

result = pipeline.run("Your draft goes here.")
print(result.final_text)
```

## Acceptance demo

Run the bundled demo to see per-pass diffs and the final rewrite on a sample draft:

```bash
python acceptance.py
```

The demo uses `MockBackend` with deterministic responses so it works offline without
any API credentials.

## Pipeline

`Pipeline` chains one or more named passes over a piece of text. Each pass sends the
current text to the configured backend and returns a `PassResult` containing the
original text, the rewrite, and a unified diff.

Available passes: `clarity`, `conciseness`, `tone`, `consistency`, `adversarial`.

```python
config = PipelineConfig(passes=["clarity", "tone"])
```

`PipelineResult.pass_results` holds one `PassResult` per pass;
`PipelineResult.final_text` is the output of the last pass.

## Backend

Two backends are provided:

| Backend | Description |
|---|---|
| `MockBackend` | Cycles through a fixed list of strings; fully deterministic, no network. |
| `ClaudeCLIBackend` | Calls the `claude` CLI via subprocess; requires the CLI to be installed. |

Implement the `complete(prompt: str) -> str` interface to add your own backend.

```python
from writing_assistant.backends import ClaudeCLIBackend

backend = ClaudeCLIBackend(model="claude-sonnet-4-6")
```

## Style profile

`StyleProfile` (in `writing_assistant/style_profile.py`) learns statistical style
features from a set of sample texts and scores how closely a new text matches them.

```python
from writing_assistant.style_profile import StyleProfile

profile = StyleProfile()
profile.fit(["Sample text one.", "Sample text two."])
score = profile.similarity_score("New text to score.")
print(score)  # 0.0–1.0
```

Features captured: average sentence length, bigram/trigram frequencies, connector
word usage, and a tone fingerprint (positivity, negativity, question density,
exclamation density, average word length).

## Running tests

```bash
pytest tests/
```
