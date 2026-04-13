# Judges

A **Judge** evaluates whether text satisfies a semantic intent. semantix ships with six judge implementations, each offering a different speed/accuracy/cost tradeoff.

## Overview

| Judge | Speed | Model size | API key | Recommended threshold |
|---|---|---|---|---|
| `QuantizedNLIJudge` | ~10ms | ~25 MB | No | 0.3 |
| `NLIJudge` | ~15ms | ~85 MB | No | 0.3 |
| `EmbeddingJudge` | ~5ms | ~80 MB | No | 0.8 |
| `LLMJudge` | ~500ms | Remote | Yes | 0.7 |
| `ForensicJudge` | Varies | Wraps any judge | Depends | Same as wrapped judge |
| `CachingJudge` | Instant (cached) | Wraps any judge | Depends | Same as wrapped judge |

## NLIJudge

The default judge. Uses a cross-encoder NLI (Natural Language Inference) model to check whether the output *entails* the intent description.

```python
from semantix import NLIJudge

judge = NLIJudge()
# or with a custom model:
judge = NLIJudge(model_name="cross-encoder/nli-MiniLM2-L6-H768")
```

**How it works:** Converts the intent description into an NLI hypothesis (e.g., "The text must politely decline..." becomes "Someone is politely declining..."), then scores the entailment probability using softmax over the cross-encoder logits. The progressive framing scores higher on NLI models because they treat the premise as evidence of an ongoing action.

**Requires:** `pip install semantix-ai` (installs `sentence-transformers`)

## QuantizedNLIJudge

INT8 quantized version of NLIJudge using ONNX Runtime. Same accuracy, ~50% faster, and no PyTorch dependency.

```python
from semantix import QuantizedNLIJudge

judge = QuantizedNLIJudge()
```

Auto-detects the best ONNX variant for your CPU architecture:

- AVX-512 VNNI
- AVX-512
- AVX2 (default x86)
- ARM64

**Requires:** `pip install "semantix-ai[turbo]"` (installs `onnxruntime`, `tokenizers`, `huggingface-hub`)

!!! tip
    When the `turbo` extra is installed, `@validate_intent` and `assert_semantic` automatically prefer `QuantizedNLIJudge` over `NLIJudge`.

## EmbeddingJudge

Uses cosine similarity between sentence embeddings. The fastest option but less accurate for nuanced intents.

```python
from semantix import EmbeddingJudge

judge = EmbeddingJudge()
# or with a custom model:
judge = EmbeddingJudge(model_name="all-MiniLM-L6-v2")
```

**How it works:** Encodes both the output and intent description using a sentence transformer, then computes cosine similarity. Good for simple semantic similarity checks, less reliable for nuanced requirements like negation or multi-part intents.

**Requires:** `pip install semantix-ai` (installs `sentence-transformers`)

## LLMJudge

Asks a lightweight LLM whether the output satisfies the intent. Most accurate, but requires an API key and is slower.

```python
from semantix import LLMJudge

judge = LLMJudge(model="gpt-4o-mini")
# or with explicit config:
judge = LLMJudge(
    model="gpt-4o-mini",
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    base_url="https://...",    # optional, for compatible endpoints
)
```

**How it works:** Sends a structured prompt asking the LLM to score the output 0.0-1.0 with a one-sentence reason. Uses `temperature=0` and `max_tokens=100` for deterministic, fast responses.

**Compatible endpoints:** Any OpenAI-compatible API -- Azure OpenAI, local vLLM, Ollama, etc. Just set `base_url`.

**Requires:** `pip install "semantix-ai[openai]"` (installs `openai`)

## ForensicJudge

Wraps any judge and adds token-level attribution on failure. On success, returns the base verdict unchanged (zero overhead).

```python
from semantix import ForensicJudge, QuantizedNLIJudge

judge = ForensicJudge(QuantizedNLIJudge())
```

```python
judge = ForensicJudge(QuantizedNLIJudge(), top_k=5)
```

**How it works:** On failure, runs mask-perturbation saliency analysis -- removes each token one at a time and measures how much the contradiction score drops. The tokens whose removal most reduces contradiction are flagged as "suspect tokens". The result is a structured Breach Report in the verdict's `reason` field:

```
## Breach Report

**Score:** 0.0512
**Base judge reason:** ...

### Token Attribution
**indemnify** (0.15), **forfeit** (0.12), **waive** (0.08)

### Summary
Intent failed. High contradiction detected. Suspect Tokens: [indemnify, forfeit, waive]
```

| Parameter | Description |
|---|---|
| `base_judge` | Any `Judge` instance to wrap |
| `top_k` | Number of suspect tokens to identify (default 3) |

## CachingJudge

LRU-cache wrapper around any judge. Caches verdicts keyed on `(output_hash, description_hash, threshold)`.

```python
from semantix import CachingJudge, NLIJudge

judge = CachingJudge(NLIJudge(), maxsize=256)
```

Useful when you're validating the same outputs repeatedly (e.g., in test suites or batch processing).

```python
# Check cache stats
print(judge.hits, judge.misses)

# Clear the cache
judge.clear()
```

| Parameter | Description |
|---|---|
| `judge` | The underlying judge to delegate to on cache misses |
| `maxsize` | Maximum number of cached verdicts (default 256, LRU eviction) |

## The Judge interface

All judges implement the same interface:

```python
from semantix.judges import Judge, Verdict

class MyCustomJudge(Judge):
    recommended_threshold = 0.7  # optional

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        # Your logic here
        return Verdict(passed=True, score=0.95, reason="Looks good")
```

The `Verdict` dataclass:

| Field | Type | Description |
|---|---|---|
| `passed` | `bool` | Whether the output satisfies the intent |
| `score` | `float \| None` | 0-1 confidence score (when available) |
| `reason` | `str \| None` | Optional explanation |

## Related

- [Intents](concepts/intents.md) -- what judges evaluate against
- [Testing](testing/assert-semantic.md) -- using judges in assertions
- [Training](training.md) -- calibrating thresholds from data
