# Where semantix fits

A grounded comparison of semantix-ai against other tools in the "validate LLM output" space. No marketing spin — the goal is to help you pick the right tool for the job, even when that tool isn't us.

## The jobs people hire validation tools to do

Different tools solve different problems. The first question is which job you have:

| Job | What you're validating | Example |
|---|---|---|
| **Shape** | Is the output a well-formed object with the right fields and types? | "Must return `{name: str, age: int}`" |
| **Content safety** | Is the output toxic, biased, or leaking PII? | "Must not contain profanity or customer emails" |
| **Grounding** | Is the output supported by the provided context? | "Answer must cite facts from the retrieved docs" |
| **Semantic intent** | Does the output *mean* what it was supposed to mean? | "Response must be polite and resolve the complaint" |
| **Consistency** | Does the output follow the prompt's constraints? | "Must stay in character, must not exceed 50 words" |

semantix-ai is built for **semantic intent** and **grounding**. If your problem is shape or content safety, other tools fit better — we'll point you to them.

## The map

### Tier 1 — direct technical overlap

These are the tools that compete for the same job as semantix on the same principles (local or NLI-based semantic evaluation).

| Tool | License | Primary primitive | Deterministic | Audit trail | DSPy native | Notes |
|---|---|---|---|---|---|---|
| **semantix-ai** | MIT | NLI-based `Intent` validation | Yes | Yes (hash-chained receipts) | Yes (`semantic_reward`, `semantic_metric`) | Local quantized model, zero API cost |
| [TruLens](https://github.com/truera/trulens) | MIT | Feedback functions (NLI + LLM-judge) | Partial | No | No (via custom glue) | Snowflake-backed, broader eval framework |
| [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) | Apache 2.0 | NLI hallucination model | Yes | No | No | Model only, not a framework |
| [DeepEval](https://github.com/confident-ai/deepeval) | Apache 2.0 | pytest-style LLM eval | Partial (LLM-judge drift) | No | No | Strong in test harness space; overlaps with pytest-semantix |

### Tier 2 — category gravity (own the keyword)

These are what people find first when they google "validate LLM output." Worth knowing even when they don't fit your job.

| Tool | What it actually does | Overlap with semantix |
|---|---|---|
| [Guardrails-AI](https://github.com/guardrails-ai/guardrails) | Shape/schema + some semantic validators, often LLM-delegated | Mostly orthogonal (shape-first); some keyword collision |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG-specific metrics (faithfulness, context recall), LLM-judge-based | Overlap on grounding; different default buyer (RAG teams) |
| [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Policy-DSL (Colang) for dialogue flow, safety, fact-checking | Different mental model; enterprise sales motion |

### Tier 3 — platforms that swallow the category

If you're already inside one of these ecosystems, the native option is the default.

- **[LangSmith](https://docs.smith.langchain.com/)** — LangChain's eval platform with built-in evaluators.
- **[Braintrust](https://www.braintrust.dev/)**, **[Humanloop](https://humanloop.com/)**, **[Arize Phoenix](https://phoenix.arize.com/)** — commercial eval platforms aimed at MLOps teams.

### Tier 4 — adjacent, not competing

Often cited in the same sentence but solve different problems.

- **[Instructor](https://github.com/jxnl/instructor)**, **[Pydantic-AI](https://ai.pydantic.dev/)** — structured output via Pydantic schemas. Shape, not meaning.
- **[Outlines](https://github.com/dottxt-ai/outlines)** — constrained generation at the token level. Prevents invalid output rather than validating it.
- **OpenAI structured outputs / Anthropic tool use** — native shape enforcement from the model provider.

## The real competition

For most teams, the default choice isn't any of the tools above. It's **calling a bigger LLM as a judge**:

```python
score = judge_lm("does this output match the intent? 0.0 to 1.0")
```

Two lines, no new dependency, "good enough" on most intents. semantix's real job is to beat this reflex where it matters — and to be honest about where it doesn't.

### Where semantix beats LLM-as-judge

- **Determinism** — same input, same score, every time. LLM-judges drift run-to-run.
- **Latency** — ~15-50 ms per call vs 300-1500 ms for an LLM-judge.
- **Cost at scale** — $0 vs ~$0.10+ per 1000 evaluations. Matters for DSPy optimization loops that call the judge thousands of times.
- **Offline / air-gapped** — runs locally, no API key, no network.
- **Audit trail** — hash-chained semantic certificates you can cite in a compliance review. No competitor in the semantic-validation category ships this.

### Where LLM-as-judge beats semantix

- **Multi-hop reasoning intents** — "is this answer compliant with section 4(b) of the policy?" Entailment models can't do this; reasoning models can.
- **World knowledge** — anything requiring facts the NLI model wasn't trained on.
- **Freeform rationales** — when you need *why*, not just a score.
- **Low-volume, high-stakes eval** — if you're calling the judge 50 times for a final go/no-go, latency and cost don't matter; reasoning quality does.

## When to pick what

| Your job | Best tool |
|---|---|
| DSPy optimization loops, pytest CI, high-volume eval | **semantix-ai** |
| RAG faithfulness / context recall, already on LangChain | **RAGAS** or **LangSmith evaluators** |
| Dialogue safety, content moderation, policy enforcement | **NeMo Guardrails** |
| Structured output with type-checked fields | **Instructor**, **Pydantic-AI**, or native provider APIs |
| Broader eval platform for an MLOps team | **TruLens**, **Braintrust**, **Humanloop**, **Phoenix** |
| Regulated industry requiring cryptographic audit trail | **semantix-ai** (ForensicJudge + AuditEngine) |
| Ad-hoc one-off evaluation in a notebook | LLM-as-judge is fine |

## What we commit to

- Honest numbers, published with methodology and raw data. See [`benchmarks/`](https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks) for reproducible comparisons.
- Updates to this page when a competitor ships something that changes the trade-offs.
- Pointers away from semantix when another tool is a better fit for your job.
