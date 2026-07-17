<p align="center">
  <h1 align="center">semantix-ai</h1>
  <p align="center"><strong>Regulator-clause-trained NLI judges, datasets, and audit trails — for compliance work in jurisdictions large vendors skip.</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/v/semantix-ai?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/pyversions/semantix-ai" alt="Python versions"></a>
  <a href="https://github.com/labrat-akhona/semantix-ai/blob/master/LICENSE"><img src="https://img.shields.io/github/license/labrat-akhona/semantix-ai" alt="License"></a>
  <a href="https://pypi.org/project/semantix-ai/"><img src="https://img.shields.io/pypi/dm/semantix-ai?color=green" alt="Downloads"></a>
  <a href="https://labrat-akhona.github.io/semantix-ai/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
  <a href="https://glama.ai/mcp/servers/labrat-akhona/semantix-ai"><img src="https://glama.ai/mcp/servers/labrat-akhona/semantix-ai/badges/score.svg" alt="semantix-ai MCP server"></a>
</p>

---

## The SA AI Compliance Stack

`semantix-ai` is the Python entry point to a coherent, Apache-licensed compliance stack built around South Africa's Protection of Personal Information Act (POPIA), the first publicly-distributed regulator-clause-fine-tuned NLI judge for any jurisdiction.

| Artifact | What it is | Where |
|---|---|---|
| **`semantix-ai`** | Decorator + library that wraps a judge around every LLM call, with hash-chained audit certificates | [PyPI](https://pypi.org/project/semantix-ai/) |
| **`nli-popia-v2`** | 10-clause POPIA-grounded NLI judge (consent, minimality, security, breach, cross-border, data-subject-rights, children, special PI, automated decision-making, general processing) | [HuggingFace](https://huggingface.co/labrat-aiko/nli-popia-v2) |
| **`sa-compliance-embeddings-v1`** | 384-dim embeddings fine-tuned on POPIA Act text + grounded scenarios — POPIA-section retrieval (recall@1 0.211 → 0.477 over `bge-small-en-v1.5`) | [HuggingFace](https://huggingface.co/labrat-aiko/sa-compliance-embeddings-v1) |
| **`popia-instruct-v0`** | QLoRA adapter on Phi-3-mini for grounded POPIA Q&A. v0 — narrow but real: clause routing + section text recitation, not free-form legal reasoning | [HuggingFace](https://huggingface.co/labrat-aiko/popia-instruct-v0) |
| **`POPIA-Bench v1`** | 197-pair public benchmark for clause-level POPIA NLI, with pinned eval hashes and a community leaderboard | [`bench/popia-v1/`](bench/popia-v1/) |
| **`POPIAJudge` preprint** | arXiv cs.CL paper documenting the recipe, results, and limitations | [`papers/popiajudge-arxiv/`](papers/popiajudge-arxiv/) |

> No competitor publicly distributes a regulator-clause-fine-tuned NLI judge. Llama Guard is trained on hazard taxonomies, Patronus Lynx on RAG faithfulness, Guardrails Hub on PII patterns + classifiers. The clause-pinned compliance niche is empty.

The library below makes the judge usable in a Python program. The stack above makes it *defensible* in a regulatory review.

---

## Quick start

Validate every LLM output against an explicit intent. Get back a score, a verdict, and a tamper-evident receipt. Locally. In ~15-50 milliseconds. Without an API key.

```bash
pip install semantix-ai
```

```python
from semantix import Intent, validate_intent

class ResolutionPolite(Intent):
    """The response must acknowledge the customer's issue and propose a concrete next step, in a polite tone."""

@validate_intent(ResolutionPolite, audit=True)
def handle_complaint(message: str) -> str:
    return call_my_llm(message)

reply = handle_complaint(incoming)
# Returns the validated reply — or raises SemanticIntentError.
# The audit engine has already written a hash-chained receipt to disk.
```

---

## Why this exists

LLM applications quietly skip the step where you prove the output was fit for purpose. The common fix — calling a bigger LLM as a judge — has three problems:

1. **It drifts.** Same input, different score on different runs. A regulator asking "rerun this validation" gets a different answer, which is indistinguishable from evidence the system is broken.
2. **It ships personal information out of your network.** Every judge call sends the output to a third-party API. Under POPIA §72 (or GDPR Art. 44, or the EU AI Act's high-risk-system obligations) that's a problem to document, not a default.
3. **It produces no receipt.** The validation happened, a score came back, nothing was recorded in a form that survives an audit.

semantix replaces that reflex with a local, deterministic validator and a tamper-evident log. Every validation produces a JSON-LD certificate hash-chained to the previous one. Modify any entry mid-chain and every subsequent hash breaks. The regulator doesn't need to trust your database — the math proves the chain is internally consistent.

---

## What you get

### 1. Validation as a decorator

```python
from semantix import Intent, validate_intent

class MedicalAdvice(Intent):
    """The text provides a medical diagnosis or treatment recommendation."""

@validate_intent(~MedicalAdvice)  # Must NOT give medical advice
def chatbot(msg: str) -> str:
    return call_my_llm(msg)
```

Compose with `&` (all must pass) and `|` (any must pass):

```python
SafeAndPolite = Polite & ~MedicalAdvice & ~LegalAdvice
```

### 2. Tamper-evident audit trail

```python
from semantix.audit.engine import AuditEngine
engine = AuditEngine()

# Bind each certificate to WHAT was judged, BY WHICH judge, ABOUT WHOM.
engine.record(
    intent="POPIA cross-border transfers",
    output=policy_text,                                  # the premise (hashed, never stored raw)
    hypothesis="Personal information is transferred outside South Africa",
    judge_id="POPIAJudge/v1@0.75",
    subject="user:5b4c9d12",
    metadata={"destination": "Ashby", "country": "US"},
    score=0.53, passed=False, reason="No consent basis on record.",
)

engine.verify_chain()   # True if no tampering
engine.chain_report()   # integrity AND variety — flags a chain that verifies
                        # perfectly while certifying one repeated result
```

Each certificate records the hash of the validated text (`output_hash`) and of the
judged claim (`claim_hash`), the intent and hypothesis, the judge identity and
configuration (`judge_id`, `metadata`), the subject, the verdict, the timestamp, and
the hash of the previous certificate. New certificates use the `…/v2` schema; existing
`…/v1` certificates still verify unchanged, so a chain that upgrades mid-life stays one
intact chain. Compatible with JSON-LD tooling and standard audit pipelines.

### 3. Self-healing retries

On failure, semantix injects structured feedback so the LLM knows what went wrong:

```python
from typing import Optional

@validate_intent(ResolutionPolite, retries=2)
def reply(msg: str, semantix_feedback: Optional[str] = None) -> str:
    prompt = f"Reply to: {msg}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_llm(prompt)
```

First call: `semantix_feedback` is `None`. On retry: it receives a Markdown report with the score, reason, and rejected output. Measured reliability improves from 21% to 70% across three intent categories.

### 4. Forensic token-level attribution

```python
from semantix import ForensicJudge, QuantizedNLIJudge
judge = ForensicJudge(QuantizedNLIJudge())
# Verdict.reason: "Suspect tokens: [indemnify, forfeit, waive]"
```

### 5. pytest integration

```python
from semantix.testing import assert_semantic

def test_chatbot_is_polite():
    response = my_chatbot("handle angry customer")
    assert_semantic(response, "polite and professional")
```

On failure:

```
AssertionError: Semantic check failed (score=0.12)
  Intent:  polite and professional
  Output:  "You're an idiot for asking that."
  Reason:  Text contains aggressive language
```

First-class pytest plugin with fixtures, markers, and CI reporting: [`pytest-semantix`](https://github.com/labrat-akhona/pytest-semantix).

---

## Framework integrations

Drop into your existing stack — retries are handled natively by each framework.

### DSPy

```python
import dspy
from semantix.integrations.dspy import semantic_reward

qa = dspy.ChainOfThought("question -> answer")
refined = dspy.Refine(module=qa, N=3, reward_fn=semantic_reward(Polite))
```

`semantic_reward` / `semantic_metric` also plug into `dspy.BestOfN`, `dspy.Evaluate`, and MIPROv2 — local, no API calls, ~15 ms per eval. See [`benchmarks/`](benchmarks/) for reproducible comparisons against LLM-judge reward functions.

<details>
<summary><strong>LangChain</strong></summary>

```python
from semantix.integrations.langchain import SemanticValidator
validator = SemanticValidator(Polite)
chain = prompt | llm | StrOutputParser() | validator
```

</details>

<details>
<summary><strong>Pydantic AI</strong></summary>

```python
from pydantic_ai import Agent
from semantix.integrations.pydantic_ai import semantix_validator
agent = Agent("openai:gpt-4o", output_type=str)
agent.output_validator(semantix_validator(Polite))
```

</details>

<details>
<summary><strong>Guardrails AI</strong></summary>

```python
from guardrails import Guard
from semantix.integrations.guardrails import SemanticIntent
guard = Guard().use(SemanticIntent("must be polite and professional"))
```

</details>

<details>
<summary><strong>Instructor</strong></summary>

```python
from semantix.integrations.instructor import SemanticStr
from pydantic import BaseModel
class Response(BaseModel):
    reply: SemanticStr["must be polite and professional", 0.85]
```

</details>

<details>
<summary><strong>MCP</strong></summary>

```bash
pip install "semantix-ai[mcp,nli]"
mcp run semantix/mcp/server.py
```

Any MCP-capable agent (Claude Desktop, Cursor, etc.) can validate intents as a tool.

</details>

<details>
<summary><strong>GitHub Actions</strong></summary>

```yaml
- uses: labrat-akhona/semantic-test-action@v1
  with:
    test-path: tests/
```

Posts a semantic test report as a PR comment.

</details>

Install extras: `pip install "semantix-ai[dspy]"`, `"[langchain]"`, `"[pydantic-ai]"`, `"[guardrails]"`, `"[instructor]"`, `"[mcp]"`, `"[all]"`.

---

## Pluggable judges

Choose the speed / accuracy / reasoning trade-off:

```python
from semantix import NLIJudge, EmbeddingJudge, LLMJudge, CachingJudge

@validate_intent(judge=NLIJudge())                           # local, ~15 ms, deterministic
@validate_intent(judge=EmbeddingJudge())                     # local, ~5 ms, similarity-based
@validate_intent(judge=LLMJudge(model="gpt-4o-mini"))        # reasoning, ~500 ms, API
@validate_intent(judge=CachingJudge(NLIJudge(), maxsize=256))  # LRU-wrapped
```

Quantized mode (INT8 ONNX, ~79 MB, no PyTorch):

```bash
pip install "semantix-ai[turbo]"
```

---

## When this is the right tool

- You're running an LLM-backed system that processes personal information and need an auditable validation step.
- You're optimising a DSPy program and the LLM-judge reward loop is too slow, too expensive, or too non-deterministic.
- You need semantic test assertions in pytest / CI that don't call a paid API.
- You're in a regulated industry (financial services, insurance, healthcare) and "the model said it was fine" isn't a defensible answer.

## When it isn't

- Your validation intent requires multi-hop reasoning or world knowledge ("is this compliant with section 4(b) of the 2026 tax code"). NLI can't do this; reasoning LLMs can.
- You need the judge to explain *why* in prose, not just give a score.
- You're evaluating fewer than 100 outputs per month and the latency / cost of LLM-as-judge doesn't matter.

See [Where semantix fits](https://labrat-akhona.github.io/semantix-ai/competitive/) for a comparison against TruLens, DeepEval, Vectara HHEM, Guardrails, RAGAS, and NeMo.

---

## Key properties

- **Local inference** — NLI model runs on CPU, no data leaves your machine.
- **Deterministic** — same input, same score, every time, on every machine. Seedable.
- **Fast** — ~15-50 ms per check with the quantized judge.
- **Zero API cost** — no tokens burned for validation.
- **Auditable** — hash-chained JSON-LD certificates per check.
- **Well-tested** — 274 tests, MIT licensed (model weights and datasets ship under Apache-2.0 / CC-BY-4.0).

---

## Installation

```bash
pip install semantix-ai                    # Core (default NLI judge)
pip install "semantix-ai[turbo]"           # Quantized ONNX (smallest footprint)
pip install "semantix-ai[openai]"          # LLM judge (GPT-4o-mini)
pip install "semantix-ai[all]"             # Everything
```

> Package name on PyPI is `semantix-ai`. Import is `from semantix import ...`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, testing, and submission guidelines.

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <em>Built by <a href="https://github.com/labrat-akhona">Akhona Eland</a> in South Africa</em>
</p>
