# POPIA and LLM outputs: what your validation trail needs to prove

**Draft — not yet published. Target venue: Dev.to + LinkedIn cross-post. Aim: show up in Google when a compliance officer or AI governance lead at an SA enterprise searches "POPIA LLM validation" or "audit LLM output South Africa."**

---

## TL;DR

- POPIA §19 requires "appropriate, reasonable technical measures" to secure personal information processed by automated systems.
- When that automated system is an LLM, "the model said it was fine" is not a technical measure.
- The defensible position: every consequential LLM output goes through a validation step, and every validation step leaves a tamper-evident receipt.
- This post walks through what that receipt needs to contain, why hash-chaining matters, and how to produce one with an open-source Python library (semantix-ai) in under 20 lines of code.

## The problem POPIA quietly creates

Section 19 of the Protection of Personal Information Act requires responsible parties to put in place "appropriate, reasonable technical and organisational measures" to secure the integrity and confidentiality of personal information. Section 71 gives data subjects the right to know the logic involved in automated decisions that significantly affect them.

Neither section mentions LLMs. Both apply to them.

If your contact-centre assistant, claims summariser, KYC reviewer, medical-aid authorisation helper, or fraud screening tool uses an LLM somewhere in the decision loop, POPIA sits over it. When a data subject invokes section 71, or the Information Regulator invokes section 81, the question you'll get asked is not "was your model good?" It's "can you demonstrate that the output was validated before it affected the customer, and can you show me the validation?"

"The model said it was fine" is not an answer. Nor is "we use GPT-4, it's generally reliable." The regulator is looking for a **process** — a reproducible, inspectable, tamper-evident record of what was checked, by what, against what criteria, and when.

## What the validation trail needs to contain

At minimum, per output, per validation:

1. **The text that was validated.** Hashed if the text itself contains PII.
2. **The intent it was validated against.** Machine-readable, not just a prose description.
3. **The validator's identity.** Model name, version, threshold, configuration.
4. **The verdict.** Pass/fail and the numeric score.
5. **A timestamp.** UTC, ISO 8601, to-the-millisecond.
6. **A link to the previous record.** This is the part most systems miss.

That last one is the difference between a log file and an audit trail. If each record contains a cryptographic hash of the previous record, the entire chain becomes tamper-evident: modify one entry and every subsequent hash breaks. The regulator doesn't need to trust your database; the math proves the record is intact.

## Why "call GPT-4 as a judge" fails the POPIA test

A common pattern is to validate one LLM with another LLM:

```python
score = judge_lm("does this response meet our quality policy? 0.0 to 1.0")
```

This works, until it doesn't. Specifically:

- **It's not deterministic.** The same input produces different scores on different runs. A regulator asking "show me this validation rerun" gets a different answer, which is indistinguishable from evidence that the system is broken.
- **It sends personal information out of your network.** Every validation call ships the LLM output — which may contain a customer's complaint, claim, or health information — to a third-party API. POPIA §72 restricts cross-border transfer of PI. Most API providers host outside South Africa.
- **It produces no receipt.** The validation happened, the score was returned, nothing was permanently recorded in a tamper-evident form.

For non-regulated, low-stakes evaluation, this pattern is fine. For anything that touches personal information under POPIA, it leaves you without a defence.

## The local, deterministic, logged alternative

Natural Language Inference (NLI) models have been quietly solving a narrower version of this problem for a decade. Give them two strings — a premise and a hypothesis — and they return the probability that the premise entails the hypothesis. Deterministically. In ~15 milliseconds. On CPU. For free.

Wrap that primitive into a validation function, pair it with a hash-chained JSON-LD receipt, and you get something POPIA can see:

```python
from semantix import Intent, validate_intent

class ClaimResolution(Intent):
    """The response must acknowledge the claim, state the decision, and give the customer their next step."""

@validate_intent(ClaimResolution, audit=True)
def summarise_claim(claim_text: str) -> str:
    return llm.complete(claim_text)

summary = summarise_claim(incoming_claim)
# The audit engine has already written a hash-chained receipt to disk.
```

Every call produces a Semantic Certificate: a signed JSON-LD record containing the hash of the input, the intent, the verdict, the timestamp, and the hash of the previous certificate. Nothing leaves your network. Nothing drifts between runs. Nothing can be silently modified.

When the Information Regulator (or your own internal audit team) asks for evidence, you hand over the chain and they can verify it end-to-end with a three-line Python script.

## Honest limits

This approach is not a silver bullet. NLI models can answer "does this text entail this statement?" well. They cannot answer "is this legally compliant under section 4 of the policy?" or "is this factually accurate about current tax law." For those intents, you still need a reasoning LLM.

The right architecture in regulated environments is usually:

1. A fast, deterministic NLI check for the structural and semantic properties you can specify precisely.
2. A human-in-the-loop review for the edge cases the NLI check flags.
3. An LLM-judge only where its reasoning adds value the NLI check can't provide — and even then, logged with the same receipt discipline.

The point is not to avoid LLMs. It's to be able to prove what happened, at what stage, with what confidence.

## Getting started

semantix-ai is MIT-licensed and available on PyPI: `pip install "semantix-ai"`.

- [Quickstart](https://labrat-akhona.github.io/semantix-ai/getting-started/)
- [Audit engine docs](https://labrat-akhona.github.io/semantix-ai/advanced/)
- [DSPy integration](https://labrat-akhona.github.io/semantix-ai/integrations/dspy/)
- [Competitive landscape](https://labrat-akhona.github.io/semantix-ai/competitive/)

If you're on an AI governance, model risk, or data protection team at a South African financial services, insurance, healthcare, or telecoms organisation and POPIA enforcement is getting real, I'd value a 30-minute conversation. Reply to this post or message me on LinkedIn.

---

*Akhona Eland is a South African engineer and the maintainer of semantix-ai. No affiliation with the Information Regulator; this post reflects my reading of POPIA and is not legal advice. Consult a qualified data protection lawyer for your organisation's specific obligations.*
