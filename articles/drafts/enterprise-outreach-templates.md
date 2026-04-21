# Enterprise outreach templates — SA beachhead

Three short email variants for cold outreach to South African enterprise AI teams. Each one leads with a concrete demo offer, not a pitch. Each assumes a warm-but-not-introduced contact (LinkedIn, a conference introduction, or a shared acquaintance).

**Before sending anything:**

1. Verify the contact's current role on LinkedIn — titles change.
2. Personalize the first line with something specific (a recent talk, post, or public initiative). Generic = trash folder.
3. Keep it short. Under 150 words. Executives read on mobile.
4. Attach nothing on first contact. Link to the one-pager on the docs site.
5. Subject line promises a concrete question, not a product.

---

## Variant 1 — Bank / Financial services AI lead

**Target roles:** Head of AI, Chief Data Officer, Head of Model Risk, Head of AI Governance.
**Target institutions:** Standard Bank Group AI, FirstRand/FNB, Absa AI, Nedbank, Capitec, Investec, Discovery Limited, Old Mutual, Sanlam, Liberty.

**Subject:** `{{FirstName}}, a question on your LLM validation audit trail`

```
Hi {{FirstName}},

Saw your {{recent post / talk / initiative}} on {{specific topic}} — quick question that might be relevant to your team.

When your LLM-powered {{product area: e.g. contact-centre assistant, claims summariser, KYC reviewer}} validates an output, what do you show an auditor if they ask you to prove it was validated?

I maintain an open-source Python library (semantix-ai, MIT) that produces hash-chained validation receipts for every LLM output — locally, deterministically, no API calls, POPIA-friendly by default. It's being evaluated for DSPy integration upstream.

I'm in {{Johannesburg / Cape Town}} on {{date range}} and would value 20 minutes to show you what the receipt looks like and hear what you'd need to see for it to matter to model-risk sign-off. No pitch, just a working demo and your feedback.

Best,
Akhona Eland
{{phone}} · labrat-akhona.github.io/semantix-ai
```

---

## Variant 2 — Insurance / Medical aid AI lead

**Target roles:** Head of Data Science, Chief Analytics Officer, Head of Claims AI, Head of Underwriting Innovation.
**Target institutions:** Discovery, Momentum, Sanlam, Old Mutual, Santam, Hollard, Clientèle.

**Subject:** `{{FirstName}}, audit-ready LLM validation for claims / underwriting workflows`

```
Hi {{FirstName}},

Quick note from a fellow SA engineer — I've been building tooling that sits exactly where I think your {{claims / underwriting / fraud}} AI team sits today: between "the model said yes" and "prove it for regulator review."

semantix-ai is an open-source Python library (MIT) that validates LLM outputs against business intents, runs locally (no PHI leaves your network), and produces a tamper-evident audit receipt for every check. It's been benchmarked against an LLM-judge baseline on DSPy optimization loops — ~20× faster, zero API cost, deterministic scores.

Would 25 minutes next {{week/month}} work for a demo? I'll come with a working notebook and the honest list of where the tool loses to a reasoning LLM, so you can decide where it fits.

Best,
Akhona Eland
labrat-akhona.github.io/semantix-ai
```

---

## Variant 3 — Telco / Industrial innovation team

**Target roles:** Head of Innovation, GM: Emerging Tech, Chief Digital Officer.
**Target institutions:** MTN, Vodacom, Telkom, Rain, Sasol Digital, Anglo American Digital.

**Subject:** `LLM output validation without an API round-trip — relevant for {{company}}?`

```
Hi {{FirstName}},

Short and relevant — {{company}}'s {{specific initiative}} looks like it's at the scale where calling OpenAI or Anthropic to grade every LLM output gets expensive fast.

I maintain semantix-ai, an open-source Python library that replaces LLM-as-judge with a local quantized NLI model. ~15 ms per evaluation, deterministic, zero cost per call, offline-capable. We're seeing ~20× latency improvement in DSPy optimization loops.

If your team is running any LLM-backed system at volume — agents, summarisation, support automation — I'd value a 20-minute conversation to understand what your validation loop looks like today and whether this fits. No pitch, just a working demo.

Best,
Akhona Eland
labrat-akhona.github.io/semantix-ai
```

---

## Follow-up cadence

- Day 0: initial send
- Day 5: one-sentence follow-up — "bumping this to the top of your inbox; happy to skip if it's not a fit"
- Day 14: final follow-up with a concrete artifact — link to a public case study, benchmark result, or a blog post. Nothing after that.

Three touches, then move on. Enterprise sales cycles are long; you're not being rejected by silence, you're being deprioritized. Come back in a quarter.

## What to do after a "yes"

1. Send a calendar link immediately. Don't email-ping-pong dates.
2. Prepare a 3-slide deck + a 5-minute live demo. No more than 8 slides total.
3. Come with a printable one-page handout. Some execs still print.
4. Ask one question before you pitch: "what does your current validation loop look like, and what hurts about it?" Listen.
5. End with a concrete next step and a date. "I'll send a pilot plan by Friday" beats "I'll be in touch."

## What to track

Keep a simple CSV: `date | company | contact | role | response | next_action | next_action_date`. Review weekly. Ten conversations beats a hundred sends.
