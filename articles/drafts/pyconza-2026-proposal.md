# PyConZA 2026 — Talk Proposal Draft

**Status:** draft, not yet submitted. Adjust title, bio, and abstract based on the actual CFP theme when it opens.

---

## Primary proposal (45-min talk)

**Title:** `Stop asking GPT-4 to grade your LLM: deterministic semantic validation in Python`

**Type:** Long talk (45 min) — or short (25 min) if that's the format on offer.

**Audience level:** Intermediate. Attendees should know Python and have touched an LLM API at least once.

**Abstract (250 words):**

Your LLM-powered app has a quiet bug. The "LLM judge" scoring your outputs — whether in CI, in a DSPy optimization loop, or in a production guardrail — gives you a different score every time you run it. It costs you ~$0.10 per thousand calls. It can't run offline, it adds 500ms to your eval loop, and when a compliance officer asks you to prove a validation happened, all you have is a log line that says "the model said 0.87."

There's a quieter, older family of models that's been solving a narrower version of this problem for years: Natural Language Inference (NLI). They answer one question — does text A entail statement B? — and they answer it deterministically, in ~15ms, on CPU, for free.

This talk walks through building a semantic validation library around a quantized NLI model, integrating it with DSPy and pytest, and adding a hash-chained audit trail that turns every validation into a cryptographically verifiable receipt. We'll cover the honest trade-offs (NLI can't do multi-hop reasoning; LLM-judges can), the POPIA/EU AI Act compliance angle that's driving regulated-industry interest, and the measured latency/cost/agreement numbers from a reproducible DSPy benchmark.

You'll leave with a clear picture of when deterministic NLI validation beats an LLM-judge, when it doesn't, and the Python-idiomatic API that makes the choice a one-liner.

**Takeaways:**

1. When LLM-as-judge is the wrong tool and what to reach for instead.
2. How to wire NLI-based validation into DSPy optimization, pytest, LangChain, and MCP.
3. The compliance trail that makes "validate, log, prove" a procurement-ready story for SA banks and insurers under POPIA.

**Outline:**

- (5 min) The LLM-judge reflex and what it's actually costing you
- (5 min) Natural Language Inference in one slide — what the model does
- (10 min) Designing an `Intent` primitive that feels Pythonic
- (10 min) Live demo: DSPy optimization loop with an LLM-judge vs. semantix — timing, cost, determinism side-by-side
- (5 min) The audit-trail story — what POPIA and the EU AI Act are actually asking for
- (5 min) Honest limits: the intents where NLI loses to a reasoning LLM
- (5 min) Q&A

**Previously presented:** No — this is a new talk.

**Speaker:**

Akhona Eland — South African engineer and maintainer of semantix-ai, an open-source Python library for semantic validation of LLM output. Builds tooling at the intersection of Python, AI, and compliance for regulated industries.

---

## Secondary proposal (lightning / 5-min)

**Title:** `A cryptographic receipt for every LLM validation, in 20 lines of Python`

**Abstract (80 words):**

LLM outputs fail silently. Validation happens silently. Compliance officers ask "how do you know that was validated?" and you wave at a log file. In five minutes, we'll show how to turn every semantic validation into a hash-chained JSON-LD receipt — tamper-evident, replayable, and compatible with your existing DSPy or pytest workflow. Built on an open-source Python library (MIT) that runs locally with no API calls.

---

## Submission checklist

- [ ] Confirm PyConZA 2026 CFP is open and deadline
- [ ] Check theme — adjust title/abstract to match if there's one
- [ ] Update speaker bio with any new credentials
- [ ] Record a 30-second video pitch if CFP asks for it
- [ ] Upload slides draft (not required at CFP, but good to have ready)
- [ ] Submit primary + secondary (many conferences accept one speaker for one slot; a lightning fallback improves odds)

## Related conferences to submit to as fallbacks

- PyCon US 2027 CFP (opens ~Sept 2026)
- EuroPython 2026 (CFP typically Jan-Feb)
- PyData Global 2026 (online, lower bar, good for practice)
- Deep Learning Indaba 2026 (African AI community, POPIA framing lands harder here)
- SACAIR (South African Conference on AI Research) — more academic but CPE for compliance crowd
