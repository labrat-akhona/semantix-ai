# Social Media Posts for semantix-ai / pytest-semantix

## Reddit r/python

**Title:** I built a pytest plugin that validates LLM outputs in 15ms — no API key, runs locally

**Body:**

I've been building LLM apps and got frustrated with testing. String matching breaks the moment your model rephrases anything. Calling GPT-4 as a judge in CI is slow and expensive.

So I built **pytest-semantix** — a pytest plugin that uses a local NLI model to check whether your LLM output actually means what you intended.

```bash
pip install pytest-semantix
```

```python
def test_chatbot_is_polite(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

- ~15ms per assertion on CPU
- No API key or network calls
- Score + reason on failure, not just pass/fail
- `--semantic-report` for a summary table
- `--semantic-report-json` for CI integration
- Negation with `~` for compliance (e.g. `~MedicalAdvice`)

On failure you get:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

Built on top of [semantix-ai](https://pypi.org/project/semantix-ai/) which also has integrations for Guardrails AI, LangChain, Instructor, Pydantic AI, and DSPy.

- PyPI: https://pypi.org/project/pytest-semantix/
- GitHub: https://github.com/labrat-akhona/pytest-semantix
- Core library: https://github.com/labrat-akhona/semantix-ai

Would love feedback. MIT licensed.

---

## Reddit r/MachineLearning

**Title:** [P] pytest-semantix: Local NLI-based testing for LLM outputs (~15ms, no API)

**Body:**

Released a pytest plugin that validates LLM outputs against natural-language intents using a local NLI model. The core insight: you can use textual entailment to check whether "Thank you for your patience" satisfies "polite and professional" — without calling another LLM.

Key design decisions:
- Uses local NLI inference (sentence-transformers) — ~15ms on CPU, zero API cost
- Also supports quantized ONNX (INT8, ~25MB vs 500MB for PyTorch)
- Pluggable judges: swap NLI for embedding similarity or LLM-as-judge
- Composable intents with `&` (all), `|` (any), `~` (not)
- Calibration from training data (midpoint between max-rejected and min-accepted scores)

The pytest plugin adds a fixture, markers, and a `--semantic-report` CLI flag.

Paper that inspired the approach: using NLI for zero-shot text classification (Yin et al., 2019).

- GitHub: https://github.com/labrat-akhona/semantix-ai
- PyPI: https://pypi.org/project/semantix-ai/

---

## Reddit r/LocalLLaMA

**Title:** Testing LLM outputs locally with NLI — no API calls, ~15ms per check

**Body:**

If you're running local LLMs and need to validate outputs programmatically, I built a tool that might help.

`pytest-semantix` lets you write semantic tests against your LLM outputs using a local NLI model. Everything runs on your machine — no cloud calls, no API keys.

```python
def test_no_medical_advice(assert_semantic):
    response = my_local_llm("my head hurts")
    assert_semantic(response, ~MedicalAdvice)  # must NOT give medical advice
```

Works great for:
- Compliance testing (no PII, no medical/legal advice)
- Tone validation (polite, professional, helpful)
- Intent verification (actually answered the question)

The NLI model is ~500MB (or ~25MB with the quantized ONNX variant). Runs on CPU in ~15ms.

`pip install pytest-semantix`

- GitHub: https://github.com/labrat-akhona/pytest-semantix

---

## Hacker News

**Title:** Show HN: pytest-semantix – Test LLM outputs with local NLI (15ms, no API key)

**Body:**

I built a pytest plugin for testing LLM outputs semantically. Instead of string matching or calling GPT as a judge, it uses a local NLI model to check whether the output entails the intent you described.

```python
def test_chatbot_is_polite(assert_semantic):
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

~15ms per check. No API key. Runs on CPU. MIT licensed.

The failure messages include score + reason:

    AssertionError: Semantic check failed (score=0.12)
      Intent:  polite and professional
      Output:  "You're an idiot for asking that."
      Reason:  Text contains aggressive language

Also has integrations for Guardrails AI, LangChain, Instructor, Pydantic AI, and DSPy.

https://github.com/labrat-akhona/pytest-semantix

---

## Twitter/X

**Post 1:**

Just shipped pytest-semantix — test your LLM outputs in pytest with a local NLI model.

~15ms. No API key. No tokens burned.

```
pip install pytest-semantix
```

```python
def test_polite(assert_semantic):
    response = my_chatbot("angry customer")
    assert_semantic(response, "polite and professional")
```

pypi.org/project/pytest-semantix/

**Post 2 (thread):**

On failure you get score + reason, not just pass/fail:

```
AssertionError: Semantic check failed (score=0.12)
  Intent: polite and professional
  Output: "You're an idiot"
  Reason: Text contains aggressive language
```

Also supports:
- @pytest.mark.semantic markers
- --semantic-report for CI summaries
- Negation (~MedicalAdvice) for compliance
- Guardrails, LangChain, Instructor, Pydantic AI, DSPy integrations
