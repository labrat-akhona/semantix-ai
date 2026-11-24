# Judges

A **Judge** evaluates whether text satisfies a semantic intent. semantix ships with six judge implementations, each offering a different speed/accuracy/cost tradeoff.

## Overview

| Judge | Speed | Model size | API key | Recommended threshold |
|---|---|---|---|---|
| `QuantizedNLIJudge` | ~15–70 ms, varies by CPU | ~79 MB INT8 | No | 0.3 |
| `NLIJudge` | Varies by CPU and text length | ~330 MB FP32 | No | 0.3 |
| `EmbeddingJudge` | ~5ms | ~80 MB | No | 0.8 |
| `LLMJudge` | ~500ms | Remote | Yes | 0.7 |
| `ForensicJudge` | Varies | Wraps any judge | Depends | Same as wrapped judge |
| `CachingJudge` | Instant (cached) | Wraps any judge | Depends | Same as wrapped judge |

## NLIJudge

The PyTorch backend, used as the default fallback when the quantized backend is unavailable. It uses a cross-encoder NLI model to check whether the output *entails* the intent description. Default resolution prefers `QuantizedNLIJudge` when its dependencies are installed.

```python
from semantix import NLIJudge

judge = NLIJudge()
# or with a custom model:
judge = NLIJudge(model_name="cross-encoder/nli-MiniLM2-L6-H768")
```

**How it works:** Converts the intent description into an NLI hypothesis (e.g., "The text must politely decline..." becomes "Someone is politely declining..."), then scores the entailment probability using softmax over the cross-encoder logits. The progressive framing scores higher on NLI models because they treat the premise as evidence of an ongoing action.

**Requires:** `pip install "semantix-ai[nli]"` (installs `sentence-transformers`)

## QuantizedNLIJudge

INT8 quantized version of NLIJudge using ONNX Runtime, with no PyTorch dependency.
Quantization can change scores and threshold decisions. Speed depends on the CPU,
selected model variant, and text length; measure on your deployment hardware.

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

**Requires:** `pip install "semantix-ai[embeddings]"` (installs `sentence-transformers`)

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

**How it works:** Sends a structured prompt asking the LLM to score the output 0.0-1.0 with a one-sentence reason. Uses `temperature=0` and `max_tokens=100`; these settings do not guarantee identical responses across calls.

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

## POPIA model versions

Install `"semantix-ai[popia]"`, then use `from semantix import POPIAJudge`.
`POPIAJudge()` currently defaults to **v1**, covering seven clauses.
`POPIAJudge(version="v2")` adds children's information, special personal information,
and automated decision-making for ten-clause coverage. Broader coverage is a tradeoff:
the published v2 card reports lower F1 on the original seven-clause holdout. Choose
the version against your own evaluation cases. See the [model card and measured
tradeoff](https://huggingface.co/labrat-aiko/nli-popia-v2).

The September 2026 local review reproduced the v1 holdout improvement, but all three
existing preset integration examples failed their expected outcomes. These presets
combine several requirements and can receive low scores on short or incomplete
evidence. Validate them with representative, independently labelled scenarios before
using their verdicts to make decisions; the holdout result does not establish preset
reliability.

### Release gate by model file

`POPIAJudge` v1 loads the same file, `onnx/model_qint8_avx512_vnni.onnx`, on every CPU. The model
repo publishes it under three names (`model_qint8_avx512.onnx`, `model_qint8_avx512_vnni.onnx`,
`model_qint8_arm64.onnx`); they are byte-identical copies. `judge.model_file` returns the file a
judge loaded.

semantix 0.3.3 and earlier chose a file from the CPU and, on Windows, Intel macOS, and AVX2-only
Linux, loaded `onnx/model_quint8_avx2.onnx`. That file scores differently and fails the release
gate. It is still published, because those versions request it by name, but this version never
selects it. Upgrading therefore changes POPIA v1 scores on those machines.

Measured on one Intel i5-11320H with the 150-pair v1 holdout at the gate's 0.5 threshold
(2026-09-15):

| Model file | Runtime | Stock F1 | POPIA v1 F1 | Release gate |
|---|---|---|---|---|
| `model_qint8_avx512_vnni.onnx` and its two copies (loaded by this version) | Windows, onnxruntime 1.24.4 | 0.517 | 0.822 | Passes |
| `model_qint8_avx512_vnni.onnx` | Linux, onnxruntime 1.18.1 | 0.517 | 0.813 | Passes |
| `model_quint8_avx2.onnx` (legacy, 0.3.3 and earlier) | Windows, onnxruntime 1.24.4 | 0.551 | 0.802 | Fails: minimality 0.708 vs stock 0.738 |
| `model_quint8_avx2.onnx` (legacy, 0.3.3 and earlier) | Linux, onnxruntime 1.18.1 | 0.551 | 0.792 | Fails: minimality |

On AVX2-only GitHub runners (2026-09-16) the qint8 file scored 0.800 against the AVX2 file's
0.773 and was about 20% faster. On those CPUs the stock baseline on the qint8 file drops to 0.246,
so the gate's margin over stock there is not evidence of fine-tune quality; read the absolute
POPIA score instead.

`semantix eval popia --all-files` scores both files on your own machine. The gate result covers
the file this version selects, and the legacy file is reported separately. Pass
`model_variant="onnx/model_quint8_avx2.onnx"` only to reproduce scores from 0.3.3 or earlier.

The single-file policy applies to v1 only. `POPIAJudge(version="v2")` and the stock
`QuantizedNLIJudge` still choose a file from the CPU, and `GDPRJudge` uses v1's file while it
falls back to the POPIA v1 weights.

Inference is local after loading. For cached operation without Hub update requests,
set `HF_HUB_OFFLINE=1` before starting Python. Scores are statistical entailment
estimates; neither a high score nor a negated low score establishes legal compliance.

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
