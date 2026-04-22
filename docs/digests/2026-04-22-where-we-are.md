# Where We Are — 2026-04-22

> **A map of the semantix-ai project, written for Akhona.**
> Read this once. Keep it next to the kettle. Update it when the shape of the project changes — not when the details do.

---

## The elevator pitch (30 seconds)

> **Semantix is a type system for what your LLM *means*, not what it *shapes like*.**
>
> Pydantic checks that the JSON your model returned has the right fields. Semantix checks that the *text inside those fields actually satisfies your intent* — like "this is a polite decline" or "this is consistent with POPIA consent." It runs locally, fast, cheap, and ships with real fine-tuned judges for the laws that matter.

If someone asks you at a party what you're building, that's the answer.

---

## The ecosystem in one picture

```
                             ┌─────────────────────────────────┐
                             │      semantix-ai (the core)     │
                             │  MIT · pip install semantix-ai  │
                             └──────────────┬──────────────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
  ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
  │    INTENT &     │              │     JUDGES      │              │ INTEGRATIONS    │
  │   DECORATORS    │              │  (the engines)  │              │  (the outlets)  │
  └─────────────────┘              └────────┬────────┘              └─────────────────┘
   Intent                                    │                        DSPy
   @validate_intent                          │                        LangChain
   AllOf/AnyOf/Not                           │                        Guardrails
   SemanticIntentError                       │                        Pydantic-AI
   assert_semantic                           │                        Instructor
                                             │                        MCP server
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      ▼                      ▼                      ▼
             ┌──────────────┐        ┌──────────────┐      ┌──────────────────┐
             │ LLM Judge    │        │ Embedding    │      │ Quantized NLI    │
             │ (any API)    │        │ Judge        │      │ (ONNX, <100ms)   │
             └──────────────┘        └──────────────┘      └────────┬─────────┘
                                                                    │
                                           ┌────────────────────────┼───────────────┐
                                           ▼                        ▼               ▼
                                    ┌─────────────┐          ┌─────────────┐  ┌─────────────┐
                                    │ POPIAJudge  │          │  GDPRJudge  │  │   (HIPAA)   │
                                    │   v0.2.0    │          │ scaffold    │  │   v0.4?     │
                                    │   LIVE      │          │   v0.2.1    │  │             │
                                    │             │          │   today     │  │             │
                                    └──────┬──────┘          └──────┬──────┘  └─────────────┘
                                           │                        │
                                           ▼                        ▼
                                 nli-popia-v1 (HF)           nli-gdpr-v1 (HF)
                                 dataset (HF)                FALLS BACK to popia
                                 Space demo (HF)             until v0.3 fine-tune
                                 macro-F1 0.813
```

Everything above is shipping code. The dashed bits (HIPAA) are strategy, not work in progress.

---

## What exists in the world right now

### On PyPI

- **`semantix-ai`** — the MIT library. Current version `0.2.0`. Installs clean with `pip install semantix-ai`; optional extras: `[openai]`, `[embeddings]`, `[nli]`, `[mcp]`, `[turbo]`, `[popia]`, `[gdpr]`, `[train]`, `[dspy]`, `[langchain]`, `[guardrails]`, `[pydantic-ai]`, `[instructor]`, `[all]`, `[dev]`, `[docs]`.

### On Hugging Face (under `labrat-aiko/`)

- **`nli-popia-v1`** — the fine-tuned model. Apache 2.0. Macro-F1 0.813 on the pinned holdout, ~70 ms per inference on CPU, four ONNX quantisation variants (AVX2, AVX-512, AVX-512-VNNI, ARM64).
- **`popia-compliance-nli`** — the dataset. Apache 2.0. Train (180 rows) + validation (120 rows) + test (150 hash-pinned rows). The test file's SHA-256 is `120e14a55bb653…44935461` and that hash is committed to the repo.
- **`popia-judge-demo`** — the Gradio Space. Seven clauses, interactive scoring, zero-PyTorch inference.

### On GitHub (`labrat-akhona/semantix-ai`)

- The source repo (MIT).
- A mkdocs-material documentation site at <https://labrat-akhona.github.io/semantix-ai/>.
- A release-gate GitHub Actions workflow that blocks any future release from regressing on any of the seven POPIA clauses by more than 0pp.
- An integration doc page for DSPy at `/integrations/dspy/` that PR #9653 in the DSPy monorepo points to.

### On Dev.to

- A published article on POPIA fine-tuning: *"I fine-tuned a model for POPIA compliance in a month. Here's the recipe."*
- A published article on the DSPy benchmark, with a transparency footnote about the v0.1.5 label-index bug that used to poison our scores.

### Open PRs in third-party repos (merge-campaign, in-flight)

| # | Repo | State |
|---|---|---|
| 1 | `punkpeye/awesome-mcp-servers` | open |
| 2 | `Hannibal046/Awesome-LLM` | open |
| 3 | `steven2358/awesome-generative-ai` | open |
| 4 | `kyrolabs/awesome-agents` | open |
| 5 | `AthenaCore/AwesomeResponsibleAI` | open |
| 6 | `Vvkmnn/awesome-ai-eval` | open |
| 7 | `wearetyomsmnv/Awesome-LLMSecOps` | open |
| 8 | `stanfordnlp/dspy` #9653 | open (resubmission, Groq-only story) |

A ninth listing — Glama's MCP registry — is pending manual action; the Dockerfile for it is already in the repo.

### On disk, not yet public

- `docs/outreach/info-regulator-letter.md` — the letter to the SA Information Regulator, written, committed, *not sent*.
- `docs/outreach/semantix-popia-onepager.md` — the one-pager that accompanies it.
- `docs/outreach/cowork-handoff.md` — a self-contained handoff doc for the desktop Claude session that will prepare the email.
- `docs/outreach/regulator-email-brief.md` — the agent-facing brief for the same task (more operational detail).

---

## Timeline — how we got here

```
2026-04-10 ── Brainstormed framework integrations spec + self-training collector
              │
2026-04-13 ── Pro-tier design spec
              │
2026-04-21 ── POPIA fine-tune landed (v0.2.0)
              │  60 seeds, 600 training rows, hash-pinned 150 eval,
              │  macro-F1 0.813 vs 0.517 stock
              │
              ── DSPy merge-campaign spec written
              │
              ── POPIA model, dataset, and Space live on HF
              │
              ── DSPy benchmark initially run, found label-index bug
              │  (semantix reading probs[2] instead of probs[1])
              │
2026-04-22 ◀── TODAY
              │
              ├─ Re-ran DSPy benchmark with v0.2.0 fix
              │   Pearson r: −0.59 → +0.60 after the bug fix
              │
              ├─ Rewrote Dev.to article to minimal Groq-only story
              │   with a transparency footnote
              │
              ├─ Filed DSPy PR #9653 with corrected benchmark
              │
              ├─ Drafted SA Information Regulator outreach letter
              │
              ├─ Scaffolded GDPR-v0 sibling model
              │   21 seeds · 7 presets · GDPRJudge with runtime fallback
              │   to POPIA weights until nli-gdpr-v1 ships in v0.3
              │
              └─ Built desktop co-work handoff for the Regulator email
```

---

## How the judges stack

A judge is anything that answers the question "does this text satisfy this intent?" with a score between 0 and 1 and a pass/fail verdict.

```
Speed  →  Fast                                          Slow
Cost   →  Cheap                                     Expensive
Needs  →  Local                                   API + money

  QuantizedNLIJudge     EmbeddingJudge        LLMJudge
  (ONNX, ~70ms)          (sentence-t5,       (OpenAI/Anthropic)
                          ~200ms)
       ▲                     ▲                    ▲
       │                     │                    │
  specific claim      fuzzy similarity      free-text reasoning
  ("consistent with     ("does this           ("explain why this
   POPIA consent")       describe X?")         fails the intent")
```

**POPIAJudge** and **GDPRJudge** are both thin subclasses of **QuantizedNLIJudge** — they just point the same inference engine at different fine-tuned weight files. That's the *velocity* of the sibling-model play: a new regulation is one `_REPO_ID =` string and a new seeds file, not a new architecture.

---

## The moat — four lines of defence

You asked yesterday whether this thing is strong as a product. It is, but only if all four of these are real:

```
            ┌───────────────────────────────────┐
            │           THE ARTEFACT            │
            │        nli-popia-v1 · 0.813 F1    │
            │            (Apache 2.0)           │
            └───────────────┬───────────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  ┌───────────┐      ┌────────────┐      ┌──────────────┐
  │ VELOCITY  │      │  DATASET   │      │  REGULATOR   │
  │           │      │            │      │              │
  │  POPIA    │      │ public 450 │      │  letter sent │
  │  GDPR  ←──┼──────│ private v2 │      │  meeting?    │
  │  HIPAA    │      │ tomorrow   │      │  feedback?   │
  │           │      │            │      │              │
  └───────────┘      └────────────┘      └──────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │       HF SEO        │
                  │  model + dataset +  │
                  │  space = 3 surfaces │
                  │  someone searching  │
                  │  for "popia LLM"    │
                  │  hits all three     │
                  └─────────────────────┘
```

- **Velocity.** Today's GDPR scaffold was the first concrete step — the recipe transfers.
- **Dataset.** Public dataset is live. A quality-v2 (private, 2×–3× the size) is the next escalation and a real product moat.
- **Regulator.** The letter is drafted and committed. Sending is the next step. Even a "noted, thank you" reply is a paper-trail win no competitor can easily reproduce.
- **SEO.** Already shipped — three HF surfaces for the single query "POPIA NLI".

The Apache 2.0 licence is *not* a weakness. Open weights + closed dataset quality + velocity + regulator paper trail is a stronger position than closed weights and nothing else.

---

## The v0.3 road

Ordered by how much work each one is, smallest first:

| Effort | Thing | Unblocks |
|---|---|---|
| ◆ low | Open a `gdpr-v0-seeds` issue inviting contributions | community moat |
| ◆◆ medium | Author 150 GDPR eval pairs, pin the hash | v0.3 fine-tune |
| ◆◆ medium | Run `scripts/expand_gdpr_seeds.py` to get ~600 training rows | v0.3 fine-tune |
| ◆◆◆ heavy | Fine-tune nli-gdpr-v1, quantise, publish | GDPRJudge becomes real |
| ◆◆ medium | Clone the POPIA Space as a GDPR Space | demo parity |
| ◆ low | Add GDPR macro-F1 to the release-gate workflow | regression protection |
| ◆◆ medium | Write the POPIA v2 (private) dataset | dataset moat |
| ◆◆◆ heavy | Engage a clinical reviewer + scope HIPAA | v0.4 |

Rule of thumb: only one ◆◆◆ task in flight at a time.

---

## What's on your desk today

Three things that don't need engineering work, just decisions.

### 1 · Hand off the Regulator email

The co-work handoff doc at `docs/outreach/cowork-handoff.md` is ready. Open the Claude Desktop app, start a new session, paste the whole file as your opening message, let the agent do preflight, review its draft, send.

### 2 · Decide whether to open the `gdpr-v0-seeds` issue

Pro: community legitimises the fine-tune corpus.
Con: commits us publicly to shipping GDPR-v1.
My read: open it. You've already committed the scaffold to master — the public has seen the direction.

### 3 · Decide whether the Dev.to POPIA article gets a follow-up

The POPIA article is live. No GDPR announcement yet — deliberately, because the weights aren't real yet. The natural moment for a GDPR post is when nli-gdpr-v1 actually ships. Do not front-run the fine-tune.

---

## Things you should be able to say at a dinner party

- "I fine-tuned an NLI model for POPIA. It runs on your laptop."
- "Macro-F1 went from 0.517 to 0.813 — that's a 29.6 percentage-point lift."
- "It's Apache 2.0 so anyone can use it commercially without paying me."
- "The moat isn't the model, it's the dataset quality, the velocity to the next regulation, and the regulator paper trail."
- "I'm writing to the Information Regulator next, not to ask for approval but to ask them to correct my framing."
- "GDPR is next. HIPAA is next-next, but I won't do it without a clinical reviewer."

---

## Things you should *not* say

- "POPIA-compliant" (anywhere, ever)
- "Endorsed by the Regulator" (they haven't, and you haven't asked them to)
- "GDPR model released" (not yet — the scaffold is live, the weights are not)
- "We" if there is not actually a team (use "I" until that changes)

---

## If you forget everything else

Remember the three links:

1. <https://huggingface.co/labrat-aiko/nli-popia-v1> — the artefact
2. <https://github.com/labrat-akhona/semantix-ai> — the code
3. <https://huggingface.co/spaces/labrat-aiko/popia-judge-demo> — the demo

Those three pages answer 80 % of the questions anyone is going to ask you about this project.

---

*Document last updated: 2026-04-22. Next update trigger: when either GDPRJudge resolves to real weights, the Regulator replies, or a v0.3 release ships — whichever is first.*
