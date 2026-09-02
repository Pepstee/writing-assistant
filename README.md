# Writing Assistant

A lightweight, multi-pass text rewriting tool with a pluggable LLM backend and statistical style analysis. Feed it a draft; it runs it through up to five sequential editing passes — clarity, tone, conciseness, consistency, and adversarial self-review — and shows a unified diff for each one.

## Installation

Python 3.10+ required. No third-party dependencies.

```bash
pip install -e .
```

## Acceptance demo

The bundled `acceptance.py` script runs all five passes on a 142-word verbose sample draft using the deterministic rule-based backend (no network, no API key needed). Every edit it shows is a real transformation made by shipped code:

```bash
python acceptance.py
```

It prints the learned style profile, the original draft, a unified diff for each pass, and the final rewrite (which it asserts is genuinely shorter and free of the wordy phrasing). The `acceptance` file in the project root declares the executable acceptance criteria, one command per line.

## CLI

```bash
# pipe text in (uses the Claude CLI backend)
echo "Your draft here." | python -m writing_assistant

# read from a file
python -m writing_assistant my_draft.txt

# use the offline rule-based backend (no credentials, deterministic)
python -m writing_assistant --backend rules my_draft.txt

# select a different Claude model
python -m writing_assistant --model claude-haiku-4-5-20251001 my_draft.txt

# use any local or provider CLI that reads prompts from stdin and writes the rewrite to stdout
python -m writing_assistant --llm-command "ollama run llama3" my_draft.txt

# choose specific passes
python -m writing_assistant --backend rules --passes clarity,conciseness my_draft.txt

# also learn a style profile from a sample file
python -m writing_assistant --backend rules --sample reference.txt my_draft.txt

# or provide explicit desired style guidance from JSON or TOML
python -m writing_assistant --profile desired-style.toml my_draft.txt
```

Output: a per-pass diff section followed by a `Final draft:` block.

## Configuring passes

Available passes: `clarity`, `tone`, `conciseness`, `consistency`, `adversarial`.

### Python API

```python
from writing_assistant.passes import CLARITY, TONE, CONCISENESS, CONSISTENCY, ADVERSARIAL
from writing_assistant.pipeline import Pipeline
from writing_assistant.llm.rule_based import RuleBasedRewriter

passes = [CLARITY, TONE, CONCISENESS, CONSISTENCY]
backend = RuleBasedRewriter()  # offline; swap in ClaudeCliLLM for model-quality rewrites
pipeline = Pipeline(passes=passes, backend=backend)

results = pipeline.run("Your draft goes here.")
for pass_obj, result in zip(passes, results):
    print(f"=== {pass_obj.name} ===")
    print(result.diff or "(no changes)")

print("Final:", results[-1].revised)
```

Each `Pass` is a plain dataclass with `name` and `instructions` fields. You can define your own:

```python
from writing_assistant.types import Pass

my_pass = Pass(
    name="formal",
    instructions="Rewrite the text in formal academic prose.",
)
pipeline = Pipeline(passes=[my_pass], backend=backend)
```

`PassRegistry` provides deterministic name lookup when an application wants to
assemble pass lists from configuration. It stores canonical `Pass` instances,
refuses duplicate owners, and reports names in sorted order. The CLI resolves its
five built-ins through the shared `BUILTIN_PASS_REGISTRY` rather than maintaining
a second pass map:

```python
from writing_assistant.passes import PassRegistry

registry = PassRegistry()
registry.register("formal", my_pass)
pipeline = Pipeline(passes=[registry.get("formal")], backend=backend)
```

### What each built-in pass does

| Pass | What it does |
|---|---|
| `clarity` | Plain language; removes jargon; every sentence easy to understand |
| `tone` | Professional, respectful tone for a general audience |
| `conciseness` | Removes filler, redundancy, and unnecessary detail |
| `consistency` | Aligns terminology, voice, and style throughout |
| `adversarial` | Surfaces weaknesses in all prior passes and rewrites to fix them |

The `adversarial` pass is special: its prompt includes the original text, all accumulated rewrites, and the current text, giving it full history to critique.

## Plugging in a custom LLM backend

Any object with a `generate(prompt: str) -> str` method works as a backend. The `LLMBackend` Protocol in `writing_assistant/types.py` documents the contract:

```python
class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...
```

### Built-in backends

**`ClaudeCliLLM`** (in `writing_assistant/llm/claude_cli.py`) — shells out to the authenticated `claude` CLI:

```python
from writing_assistant.llm.claude_cli import ClaudeCliLLM

backend = ClaudeCliLLM(model="claude-sonnet-4-6")
pipeline = Pipeline(passes=[CLARITY, CONCISENESS], backend=backend)
```

Requires the [Claude Code CLI](https://claude.ai/code) to be installed and authenticated.

**`RuleBasedRewriter`** (in `writing_assistant/llm/rule_based.py`) — a deterministic, fully offline rule engine. It is not a language model: it detects which pass is asking from the prompt and applies that pass's editing rules (plain-language substitutions, filler removal, contraction expansion, typographic normalisation). Useful when you have no network, no credentials, or need byte-for-byte reproducible output:

```python
from writing_assistant.llm.rule_based import RuleBasedRewriter

backend = RuleBasedRewriter()
```

**`CommandCliLLM`** (in the existing CLI-backend owner,
`writing_assistant/llm/claude_cli.py`) — runs any explicit argument-vector command
without a shell, writes the exact prompt to stdin, and uses non-empty stdout as the
rewrite:

```python
from writing_assistant.llm import CommandCliLLM

backend = CommandCliLLM(["ollama", "run", "llama3"])
```

The CLI accepts the same capability through `--llm-command`. The command may also
come from `WRITING_ASSISTANT_LLM_COMMAND`; the recovered `REWRITER_LLM_COMMAND`
name remains supported for compatibility. Supplying a command selects this backend
and overrides `--model`. `--backend command` makes that choice explicit, while
`--backend rules` remains strictly offline and refuses a simultaneous command.

### Custom backend example

```python
import anthropic
from writing_assistant.pipeline import Pipeline
from writing_assistant.passes import CLARITY, TONE

class AnthropicSDKBackend:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def generate(self, prompt: str) -> str:
        message = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

pipeline = Pipeline(passes=[CLARITY, TONE], backend=AnthropicSDKBackend())
results = pipeline.run("Your draft.")
print(results[-1].revised)
```

Any provider works — OpenAI, local Ollama, a REST API — as long as the object exposes `generate`.

## Building a style profile from sample text

`StyleProfile` (in `writing_assistant/style.py`) analyses sample texts and extracts statistical style features. When passed to the `Pipeline`, the **consistency pass** receives the profile summary in its prompt so the LLM can match the target style.

```python
from writing_assistant.style import StyleProfile
from writing_assistant.pipeline import Pipeline
from writing_assistant.passes import CONSISTENCY
from writing_assistant.llm.rule_based import RuleBasedRewriter

# Learn from one or more reference texts
with open("reference.txt") as f:
    sample = f.read()

profile = StyleProfile.learn([sample])
print(profile.summary())
# Average sentence length: 12.4 words
# Passive voice ratio: 18.2%
# Vocabulary richness (type-token ratio): 0.61
# Preferred transition words: also, furthermore, however, therefore

# Use the profile in a pipeline
backend = RuleBasedRewriter()
pipeline = Pipeline(passes=[CONSISTENCY], backend=backend, style_profile=profile)
results = pipeline.run("Your draft.")
```

Pass multiple samples to `learn()` for a richer profile:

```python
profile = StyleProfile.learn([text1, text2, text3])
```

The profile captures:
- Average sentence length (words per sentence)
- Passive voice ratio
- Vocabulary richness (type-token ratio)
- Preferred transition words (`however`, `therefore`, `moreover`, …)

Save and reload a profile as JSON:

```python
json_str = profile.to_json()
restored = StyleProfile.from_json(json_str)
```

## Loading an explicit desired style

Sample-derived evidence and operator-authored preferences remain separate. Use
`DesiredStyleProfile` when you want to state the target tone, formality,
vocabulary, and sentence-length bounds directly:

```toml
tone = "friendly"
formality = "semiformal"
vocabulary = ["plain", "specific"]
min_sentence_words = 5
max_sentence_words = 24
```

```python
from writing_assistant.style import DesiredStyleProfile

desired = DesiredStyleProfile.from_file("desired-style.toml")
pipeline = Pipeline(passes=[CLARITY, TONE], backend=backend, style_profile=desired)
```

The CLI accepts the same file through `--profile`. JSON is also supported.
Unknown fields and invalid values fail before any rewrite backend is called.
`--profile` and the sample-derived `--sample` option are intentionally mutually
exclusive so declared preferences are never mislabeled as learned evidence.

## Scoring how well a text matches a style

A second, finer-grained analyser lives in `writing_assistant/style_profile.py`. Where `style.StyleProfile` produces a human-readable summary for the consistency pass prompt, `style_profile.StyleProfile` builds a statistical fingerprint (bigram/trigram frequencies, connector usage, tone markers) and scores how closely any text matches it:

```python
from writing_assistant.style_profile import StyleProfile

fingerprint = StyleProfile().fit([reference_text])
score = fingerprint.similarity_score(candidate_text)  # 0.0 – 1.0
```

Use it to measure whether a pipeline rewrite drifted away from the reference style — for example, asserting `similarity_score` stays above a threshold after the consistency pass.

## Running the test suite

```bash
pytest tests/
```

The suite covers the pipeline, all five passes, both shipped backends (the rule engine directly; the Claude CLI backend with its subprocess boundary patched), style profile logic, CLI exit codes, and the acceptance script. No network calls are made: pipeline-level tests inject a scripted test double that lives in `tests/mock_llm.py` and is never shipped.
