# Pitch Deck, Branding & LinkedIn — Desktop Co-Work Brief

> **Paste this entire document into a fresh Claude Desktop co-work session as your opening message.** Everything the agent needs is inline. The output of this session is a pitch deck, a visual identity, a branded LinkedIn post series, and an SEO playbook — all for a real, shipping open-source project called `semantix-ai`.
>
> Read this whole document before producing anything.

---

## 0 · Who you are, who you are working for, what we are doing

You are a Claude agent collaborating with **Akhona Eland**, a South African software engineer who has spent the last month building `semantix-ai`: a semantic type system for LLM outputs, with a shipping fine-tuned judge for the South African POPIA privacy law. He is one person. The work is real, measurable, and public. Over the next few weeks he wants the work to reach the people who can accelerate it — framework maintainers, AI safety researchers, privacy-tech VCs, compliance engineering leads at SA banks, and the EU GDPR + US HIPAA communities that will consume the next sibling models.

Your job in this session is to produce **positioning materials** — a pitch deck, a visual identity, a LinkedIn post series, and an SEO playbook — that cast a shadow proportionate to where this project is *going*, backed by the demonstrable progress it has *already made*.

This is a specific kind of craft. It is not marketing spin. It is the act of framing real work so that its trajectory is visible to a first-time viewer. You will hold that line carefully, and you will refuse to cross it. Section 6 tells you where the line is.

---

## 1 · Factual ammunition (everything you are allowed to cite)

Every number, link, and claim below has been verified. You may use any of them verbatim. Do not embellish.

### 1.1 · The shipping library

- **Name:** `semantix-ai`
- **Licence:** MIT
- **Install:** `pip install semantix-ai`
- **Current version:** `0.2.0`
- **Repo:** <https://github.com/labrat-akhona/semantix-ai>
- **Docs site:** <https://labrat-akhona.github.io/semantix-ai/>
- **Optional extras:** `[openai]`, `[embeddings]`, `[nli]`, `[mcp]`, `[turbo]`, `[popia]`, `[gdpr]`, `[train]`, `[dspy]`, `[langchain]`, `[guardrails]`, `[pydantic-ai]`, `[instructor]`

### 1.2 · The fine-tuned model (the single biggest flex)

- **Model ID:** `labrat-aiko/nli-popia-v1` on Hugging Face
- **URL:** <https://huggingface.co/labrat-aiko/nli-popia-v1>
- **Licence:** Apache 2.0
- **Task:** Cross-encoder NLI fine-tuned for 7 canonical POPIA clauses
- **Headline metric:** macro-F1 of **0.813** on a hash-pinned 150-pair holdout vs **0.517** for the stock NLI baseline → **+29.6 percentage points**
- **Inference speed:** ~70 ms per check on CPU (no GPU, no API call, no network)
- **ONNX variants:** four (AVX2, AVX-512, AVX-512-VNNI, ARM64)
- **Release-gate:** a GitHub Actions workflow blocks any future release from regressing on any of the seven clauses

### 1.3 · The training + evaluation dataset

- **Dataset ID:** `labrat-aiko/popia-compliance-nli`
- **URL:** <https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli>
- **Licence:** Apache 2.0
- **Splits:** train (180 hand-authored + paraphrased), validation (120 paraphrases), test (150 hash-pinned)
- **Reproducibility hook:** the SHA-256 of the test file is committed to the repo; any change to the eval must be a standalone commit

### 1.4 · The interactive demo

- **Space:** `labrat-aiko/popia-judge-demo`
- **URL:** <https://huggingface.co/spaces/labrat-aiko/popia-judge-demo>
- **Stack:** Gradio on top of `onnxruntime` and `tokenizers` — zero PyTorch at runtime
- **What it does:** a user pastes a signup flow, breach notification, or policy excerpt; the Space returns a per-clause confidence score in under a second

### 1.5 · The seven POPIA clauses

| # | Clause | POPIA anchor |
|---|---|---|
| 1 | Consent | s.11(1)(a) |
| 2 | Minimality / purpose limitation | s.10, s.13 |
| 3 | Security safeguards | s.19 |
| 4 | Breach notification | s.22 |
| 5 | Cross-border transfers | s.72 |
| 6 | General lawful processing | s.9 |
| 7 | Data subject rights | s.23–s.25 |

### 1.6 · The GDPR sibling (shipping in v0.3)

- **Status as of 2026-04-22:** scaffold committed (21 hand-authored seeds, 7 Intent presets, `GDPRJudge` class with runtime fallback to POPIA weights until `labrat-aiko/nli-gdpr-v1` is published)
- **Recipe:** identical to POPIA — 60+ seeds → LLM paraphrase → fine-tune → hash-pin → release-gate
- **Target ship:** v0.3 release, weeks not months
- **Coverage:** 7 canonical GDPR clauses (Consent, Minimality, Security, Breach, Transfers, General Processing, Data Subject Rights) keyed to specific Articles (Art. 5, 6, 7, 15–22, 32, 33–34, 44–49)

### 1.7 · Framework integrations (already shipping)

Installable today as `pip install semantix-ai[<name>]`: **DSPy**, **LangChain**, **Guardrails**, **Pydantic-AI**, **Instructor**, **MCP server**. An open PR against `stanfordnlp/dspy` (#9653) adds the library to the official DSPy community use-cases page.

### 1.8 · Articles published on Dev.to (verified via Dev.to API, 2026-04-22)

Akhona's Dev.to username: `akhona_eland_072dac9e0c2c`. All articles below are live.

| Published | Title | URL | Views | Reactions |
|---|---|---|---|---|
| 2026-04-22 | A 70ms Local NLI Judge Hits 0.596 Pearson r With Groq Llama 3.3 70B on DSPy Reward Scoring | <https://dev.to/akhona_eland_072dac9e0c2c/a-70ms-local-nli-judge-hits-0596-pearson-r-with-groq-llama-33-70b-on-dspy-reward-scoring-1d76> | 11 | 0 |
| 2026-04-13 | Build LLM Guardrails in 3 Lines of Python (No API Key, No Cloud) | <https://dev.to/akhona_eland_072dac9e0c2c/build-llm-guardrails-in-3-lines-of-python-no-api-key-no-cloud-5amf> | 2 | 0 |
| 2026-04-13 | Test Your LLM Outputs in pytest (15ms, No API Key) | <https://dev.to/akhona_eland_072dac9e0c2c/test-your-llm-outputs-in-pytest-15ms-no-api-key-1mmj> | 36 | 0 |
| 2026-04-10 | How to Fine-Tune GPT-4o-mini on Your Own Guardrail Failures (50 Lines of Python) | <https://dev.to/akhona_eland_072dac9e0c2c/how-to-fine-tune-gpt-4o-mini-on-your-own-guardrail-failures-50-lines-of-python-3l4n> | 22 | 0 |
| 2026-04-10 | Your AI Guardrail Is a Dead End. Ours Is a Feedback Loop. | <https://dev.to/akhona_eland_072dac9e0c2c/your-ai-guardrail-is-a-dead-end-ours-is-a-feedback-loop-4n6a> | 3 | 0 |
| 2026-04-06 | Escaping Pilot Purgatory: How Semantix-ai v0.1.5 Built the Immutable Trust Layer for AI Agents | <https://dev.to/akhona_eland_072dac9e0c2c/escaping-pilot-purgatory-how-semantix-ai-v015-built-the-immutable-trust-layer-for-ai-agents-a81> | 21 | 0 |
| 2026-04-06 | Any AI Agent Can Now Vibe Check LLM Outputs — No Code Required | <https://dev.to/akhona_eland_072dac9e0c2c/any-ai-agent-can-now-vibe-check-llm-outputs-no-code-required-19ei> | 49 | 1 |
| 2026-04-05 | Your LLM Passes Type Checks but Fails the "Vibe Check": How I Fixed AI Reliability | <https://dev.to/akhona_eland_072dac9e0c2c/your-llm-passes-type-checks-but-fails-the-vibe-check-how-i-fixed-ai-reliability-38ac> | 30 | 2 |
| 2026-04-02 | Your LLM Passes Type Checks but Fails the Vibe Check — Here's How to Fix It | <https://dev.to/akhona_eland_072dac9e0c2c/your-llm-passes-type-checks-but-fails-the-vibe-check-heres-how-to-fix-it-1dkm> | 24 | 0 |

**Best performer so far:** the 2026-04-06 "Any AI Agent Can Vibe Check" post at 49 views / 1 reaction. **Cumulative views across all articles ≈ 198**. This is a low floor — the launch campaign in §4.4 is the work that drives these up.

**Not yet published:** a POPIA-fine-tune explainer article exists as `articles/devto-popia-finetune.md` in the repo (untracked). The brief below (§4.4) assumes this article ships in the first week of the LinkedIn campaign.

### 1.9 · Regulator engagement

On 2026-04-22 Akhona sent a formal introduction letter to the South African Information Regulator (Adv. Pansy Tlakula, Chairperson) — not asking for endorsement, asking for framing review and clause-hypothesis feedback. Very few open-source maintainers in the privacy-tech space have written to their regulator in this framing. **Do not mention this outreach in any public deliverable until 2026-06-06 or until the Regulator replies, whichever is sooner.** See Section 6 for the full rule.

### 1.10 · Eight awesome-list PRs in-flight

Open PRs to get semantix-ai listed in community catalogues: `awesome-mcp-servers`, `Awesome-LLM`, `awesome-generative-ai`, `awesome-agents`, `AwesomeResponsibleAI`, `awesome-ai-eval`, `Awesome-LLMSecOps`, and the DSPy PR mentioned above.

### 1.11 · Current traction numbers (verified 2026-04-22)

Use these for slide 9 of the pitch deck. Do not round.

| Surface | Metric | Value |
|---|---|---|
| GitHub `labrat-akhona/semantix-ai` | Stars | 1 |
| GitHub | Forks | 0 |
| GitHub | Open issues | 0 |
| GitHub | Repo age | **26 days** (created 2026-03-27) |
| PyPI `semantix-ai` | Downloads last day | 34 |
| PyPI | Downloads last week | 121 |
| PyPI | Downloads last month | **1,391** |
| HF `labrat-aiko/nli-popia-v1` | Downloads | 12 |
| HF `labrat-aiko/popia-compliance-nli` | Downloads | 0 (published today) |
| HF `labrat-aiko/popia-judge-demo` | Likes | 0 |
| Dev.to | Published articles | 9 |
| Dev.to | Cumulative views | ≈ 198 |

**The flex-worthy number is the PyPI one.** 1,391 monthly downloads in a 26-day-old repo with 1 GitHub star is a real signal — it means the package is being found through PyPI search and install chains, not vanity stars. For the deck, frame it as "1,391 monthly PyPI installs on a repo that is 26 days old."

**The honest-but-small numbers** (GitHub stars, Dev.to reactions, HF likes) must also appear on slide 9 so the whole picture is visible. Do not cherry-pick. The launch campaign is the thing that moves them.

---

## 2 · The positioning strategy

### 2.1 · The one-line pitch (anchor everything else to this)

> **Semantix is a type system for what your LLM *means*, not just what it *shapes like*. It ships with fine-tuned judges for the privacy laws that matter — POPIA live, GDPR in weeks, HIPAA next.**

That sentence carries the full strategy: (a) the core insight (semantic vs. structural validation), (b) the moat (fine-tuned judges, not wrappers), (c) the trajectory (three regulations, roadmap visible).

### 2.2 · The three moats, ranked

1. **Velocity.** The POPIA → GDPR → HIPAA recipe is the same codepath. A competitor with deeper pockets would still need the recipe, the datasets, and the release-gate discipline. Each new regulation is a ~30-day sprint, not a ~6-month project.
2. **Dataset quality.** The POPIA dataset is public and Apache-licensed. The training-data *construction skill* is not — hand-authored seed distribution across SA provinces and business types, hash-pinned eval, incident-driven paraphrase. A private v2 dataset is trivially an option and would be a pure product moat when/if that becomes a commercial step.
3. **Regulator paper trail.** [Redacted in public materials until 2026-06-06 — internal only for this session.] Very few maintainers have a live, formal dialogue channel open with a data-protection regulator. Once it is public, this is a durable legitimacy signal no competitor can retroactively create.

### 2.3 · Flex responsibly — the "shadow bigger than yourself" rule

Ambitious framing is earned when the work is real and the trajectory is named. This project qualifies on both counts. Your framing budget:

| You may | You may not |
|---|---|
| Call it "the semantix-ai ecosystem" (it is — library + model + dataset + Space + docs + CI) | Call it "a team" or use "we" to imply employees |
| Say "shipping GDPR in Q2 2026" (the scaffold is live, the fine-tune is roadmapped) | Say "GDPR released" (the weights are not public yet) |
| Describe POPIA as "the first of seven planned compliance judges" | Claim "industry-standard" or "used by Fortune 500" — not yet |
| Use "+29.6 percentage-point macro-F1 lift" (measured, on a pinned set) | Round to "98% accurate" or similar collapsing |
| Frame the roadmap confidently — "POPIA, then GDPR, then HIPAA" | Pre-announce features that have not even been scaffolded |
| Call Akhona a "builder of privacy-tech infrastructure" | Call him a "founder" if there is no incorporated entity (ask before using that word) |
| Say "open-source under MIT / Apache 2.0 and free to use commercially" | Say "free forever" — never lock future decisions |

The rule: **project forward, never fabricate backward.** Every claim about present state must be literally true; claims about trajectory must name a roadmap that already has concrete work behind it.

---

## 3 · The audiences — who we are writing for (big ears)

Rank-order of people whose attention would materially accelerate the project:

| Tier | Who | What lands with them |
|---|---|---|
| **A** | DSPy core team (Omar Khattab, the Stanford lab) | Rigorous benchmark + the PR #9653 narrative |
| **A** | Hugging Face community leads (Omar Sanseviero, Philipp Schmid) | HF-native stack — model, dataset, Space, Apache 2.0. Good candidate for the "Daily Papers" / "Community highlight" surface |
| **A** | Clem Delangue (HF CEO) | The "local + private + compliance" story is on his public interests |
| **A** | AI safety / alignment researchers at Anthropic, OpenAI, DeepMind | The label-index-bug transparency story is a credibility signal for this audience |
| **B** | SA bank & fintech CTOs (Standard, Absa, FNB, Nedbank, Capitec, Tyme, Discovery) | POPIA is non-negotiable for them. Local inference is compliance-friendly |
| **B** | Privacy-tech VCs (Mozilla Ventures, IA Ventures, Privitar / OneTrust / Didomi alumni networks) | The moat structure + roadmap is a Series-A thesis shape |
| **B** | EU GDPR community (Max Schrems / noyb circle, Brave-style privacy folks) | The GDPR sibling and the "runs on your laptop" framing |
| **B** | LangChain + LlamaIndex + CrewAI maintainers (Harrison Chase, Jerry Liu, Joao Moura) | Integrations already shipping; they link good integrations |
| **C** | South African tech ecosystem press (Ventureburn, Techcentral, Stuff South Africa) | The "SA engineer builds NLI model for SA law" angle |
| **C** | LinkedIn AI-product community at large | General reach & recruiting signal |

Tier A is the audience every LinkedIn post and every pitch slide should be secretly writing to. Tier B and C come along for the ride if Tier A lands.

---

## 4 · Deliverables

Produce **all** of the following in this session. Organise your response as a single numbered delivery with clear headers. If you run out of context budget, prioritise in this order: 4.5 > 4.4 > 4.1 > 4.2 > 4.3.

### 4.1 · The pitch deck (12 slides)

Audience: a VC, a framework maintainer, or a CTO who has 90 seconds. Must work at both a 90-second flip-through *and* a 10-minute walk-through.

Slide order:

1. **Title.** Project name + one-line pitch + author's name + three links (GitHub, HF org, docs).
2. **The problem in one picture.** Pydantic validates shape, nothing validates meaning. A passing JSON schema can still contain a prompt injection, a leaked secret, a rude reply, or a POPIA violation.
3. **The insight.** Structural validation ≠ semantic validation. We need a type system for meaning.
4. **What semantix-ai is.** Decorator + Intent + Judge. Show the 5-line code example (write one; make it genuinely short).
5. **The moat: fine-tuned judges.** POPIA judge scored 0.813 macro-F1 vs 0.517 stock NLI. Show the delta as a bar chart.
6. **Seven POPIA clauses, on-device, in 70ms.** The table from Section 1.5 + the latency number.
7. **The recipe transfers.** POPIA shipped. GDPR scaffolded. HIPAA roadmapped. Show as a Gantt-style timeline.
8. **Ecosystem surface.** PyPI package, HF model, HF dataset, HF Space, 6 framework integrations, release-gate CI. Show as a hub-and-spoke diagram.
9. **Traction.** Pick three: PyPI download curve if you can find it, HF downloads on the model, GitHub stars, DSPy PR #9653 open, 8 awesome-list PRs in-flight. Do not invent numbers you cannot verify.
10. **Roadmap.** v0.3 (GDPR fine-tune), v0.4 (HIPAA scoping + clinical reviewer engagement), v0.5 (private dataset / commercial tier question). Name real months.
11. **Who's building it.** One slide for Akhona. Real bio. Links. Do not inflate. Leave room for "looking for collaborators in [EU GDPR, US HIPAA]" as a call-to-action.
12. **Ask.** Tailored per audience: for VCs, "we're not raising yet, we're mapping the Series-A thesis"; for framework maintainers, "list us, review the PR"; for CTOs, "try the demo, tell us where it breaks."

Output format for the pitch deck: **Markdown with one H2 per slide, followed by a bullet list of what goes on that slide, followed by a "Visual:" line describing the diagram or image, followed by a "Speaker notes:" paragraph.** The slides themselves will be rendered later in Keynote / Slides / Gamma; your job is the content spec.

### 4.2 · The visualizations

Produce **five** concrete, describable visuals. For each: (a) a one-paragraph description of what it shows, (b) the data points to include, (c) a rendering recommendation (bar chart, radar chart, timeline, architecture diagram, etc.), (d) an ASCII mock if the visual is a diagram (not a chart).

1. **The F1 bar chart.** Stock NLI 0.517 vs semantix POPIA 0.813. Annotate the +29.6pp delta.
2. **The latency chart.** semantix (70ms, local) vs typical LLM-as-judge (1–3s, cloud) vs typical keyword regex (fast but brittle).
3. **The three-judge stack.** LLM judge ← Embedding judge ← Quantized NLI judge (what trades off).
4. **The moat diagram.** The four-pillar diagram from the internal digest (velocity, dataset, regulator, SEO) redrawn for external use — drop the "Regulator" pillar and replace with a placeholder labelled "Ecosystem trust" until 2026-06-06.
5. **The recipe-transfer timeline.** POPIA shipped → GDPR scaffolded → HIPAA roadmapped as a horizontal Gantt.

### 4.3 · Branding

Produce a one-page brand brief:

- **Wordmark direction.** Two or three concrete options — typography pairing recommendations, not logos you try to draw in ASCII. Name the typefaces.
- **Colour palette.** Four to six colours with hex codes. One "trust" colour (dark, formal), one "action" colour (bright, confident), two neutrals. Name the palette (e.g. "Cape Dusk," "Savanna Sunrise" — feel free to lean into an SA-rooted identity; it is an honest part of the story).
- **Voice & tone rules.** One paragraph on the voice (I would suggest: precise, understated, confident, no hyperbole). Five "say / don't say" pairs.
- **Naming conventions.** How to refer to the project in running prose, in code blocks, in speech. (Is it "Semantix"? "semantix-ai"? "semantix"? Pick and explain.)
- **Tagline shortlist.** Three taglines, ranked by recommendation, each with one-sentence rationale.

### 4.4 · The LinkedIn post series (8 posts)

A connected mini-campaign running over ~4 weeks, designed to build narrative momentum. For each post: title/hook, full body (max 1,800 chars), recommended visual (reference the Section 4.2 visuals where possible), hashtag list, and target publish day-of-week.

1. **Launch post.** "I shipped an open-source POPIA compliance judge. 0.813 macro-F1. Runs on your laptop." Pin this one.
2. **The technical flex.** The label-index-bug transparency post. *"Here's the bug I shipped in v0.1.5 and caught in v0.2.0. This is what 'show your working' looks like in public."* This is counterintuitively strong — it builds trust faster than a pure brag.
3. **The recipe post.** "How to fine-tune an NLI model for a regulatory framework in 30 days." Carousel format ideal — one clause per slide.
4. **The GDPR announcement.** "POPIA shipped. GDPR scaffolded today. Here's the roadmap." Tease the sibling without over-promising. Must explicitly say "scaffold, not weights."
5. **The integrations post.** "semantix-ai now ships with DSPy, LangChain, Guardrails, Pydantic-AI, Instructor, and MCP. One decorator, five ecosystems." Good one to tag framework founders.
6. **The question post.** "If you're a compliance engineer in SA or EU — what's the regulation that would unlock the biggest workflow for you if it had a local, Apache-2.0 judge?" Engagement bait that also generates the v0.4 roadmap.
7. **The "why South Africa" post.** "People keep asking why I built for POPIA first. Short answer: because I live here, because the law is already in force, and because the first jurisdiction is always hardest — everything after is recipe." Humanising post.
8. **The opinion piece.** "Compliance-as-a-service is the wrong frame. Compliance judges should run on your laptop, be open-source, and belong to the developer — not rented from a vendor." Polarising, deliberate. Save for end of campaign.

SEO note for LinkedIn: LinkedIn's own search is the second-biggest professional-search surface after Google. Treat post titles and first 200 characters as search-indexed copy. Include the phrases from Section 4.5.3 naturally.

### 4.5 · SEO playbook

#### 4.5.1 · Keyword anchor set

Three tiers of phrases. Work them into titles, H1s, model card, README, LinkedIn posts, and article leads — naturally, never stuffed.

- **Tier 1 (high-intent, low-competition; own these):** `POPIA compliance AI`, `POPIA LLM validation`, `POPIA NLI model`, `GDPR compliance judge`, `local compliance validation`, `on-device privacy validation`, `compliance semantic validation`
- **Tier 2 (topical authority):** `LLM output validation`, `semantic type system`, `LLM guardrails open source`, `NLI cross-encoder fine-tuning`, `Pydantic for LLM meaning`, `ONNX inference compliance`
- **Tier 3 (broad but adjacent):** `LLM safety`, `AI compliance`, `AI guardrails`, `privacy-preserving AI`, `data protection AI`

#### 4.5.2 · Surface-by-surface checklist

For each surface below, produce a one-sentence diagnostic + recommended action:

- GitHub repo description + topics
- PyPI long description
- Hugging Face model card (first paragraph; first 160 characters are the HF social-card snippet)
- Hugging Face dataset card
- Hugging Face Space "About" (short_description field)
- docs site landing page (`/`) — H1 + meta description
- Dev.to article titles + canonical tags
- LinkedIn profile headline + "About"

#### 4.5.3 · Internal link graph

Recommend the minimum set of cross-links that make each surface rank for its own anchor phrase:

- Every HF surface links to all three other HF surfaces + the GitHub repo
- GitHub README links to all three HF surfaces + the docs site + Dev.to articles
- Dev.to articles cross-link each other and link to the HF demo Space (hands-on beats prose)

---

## 5 · Voice rules (apply to every artefact you produce)

- **Precise over breathless.** "0.813 macro-F1 on a hash-pinned holdout" beats "state-of-the-art results."
- **Understated over clever.** Most technical audiences are allergic to clever. The work is the flex.
- **First-person singular, honest.** Akhona is one person today. "I built" beats "we built." Do not hide the solo-builder fact — that *is* part of the story.
- **Show working.** Numbers come with denominators. Benchmarks come with the eval set's SHA. Claims come with the file path.
- **Shrink the distance between reader and product.** Every surface should have a "try it" link within one scroll. For most posts, that's the HF Space.

---

## 6 · Hard don'ts (non-negotiable)

1. **Do not mention the Regulator outreach until 2026-06-06.** No reference to the letter, the send, the three addresses, the Chairperson's name, or the existence of the outreach in any post, slide, branding asset, or SEO copy produced in this session. This is a hard rule — it is under a 45-day publication silence.
2. **Do not use the phrase "POPIA-compliant."** Anywhere. The model is a semantic-consistency tool, not a compliance determination. This rule carries across to GDPR and any future jurisdiction.
3. **Do not inflate headcount.** Akhona is the author. If you need a collective noun, use "the project," "the ecosystem," or "contributors" (the last one is honest because the public seed-contribution issue invites them).
4. **Do not claim endorsements that do not exist.** Not from the Regulator, not from Anthropic, not from HuggingFace, not from any framework maintainer. Open PRs are open PRs; merges are merges; cite only what is final.
5. **Do not forward-announce.** You may say "GDPR shipping in v0.3" because the scaffold is live. You may not say "HIPAA shipping in Q4" — that regulation has no scaffold yet.
6. **Do not use "Fortune 500," "industry-standard," or "trusted by leading enterprises"** — none of it is earned yet. When in doubt, delete the adjective.
7. **Do not round.** 0.813 is not 0.81 and never 0.8. +29.6 is not +30. The precision *is* the credibility signal.
8. **Do not produce anything that contradicts `docs/outreach/info-regulator-letter.md`.** If you cannot see that file, take the conservative read: semantix does not claim to determine compliance.

If you find yourself about to violate one of these, stop and replace the sentence.

---

## 7 · Return format

Produce one document per deliverable, in this order, with explicit horizontal rules between them:

1. `# Pitch deck — 12 slides` (markdown as specified in 4.1)
2. `# Visualizations — five assets` (markdown as specified in 4.2)
3. `# Branding brief` (markdown as specified in 4.3)
4. `# LinkedIn series — 8 posts` (markdown as specified in 4.4)
5. `# SEO playbook` (markdown as specified in 4.5)
6. `# Open questions for Akhona` — any decision points you surfaced that he needs to resolve before publication (e.g. choice between two tagline finalists, choice of wordmark typeface, date-cadence for posts)

Keep each deliverable self-contained so Akhona can paste any single one into the relevant downstream tool (Keynote, Figma, Scheduler) without having to strip context.

---

## 8 · One final instruction

Akhona explicitly asked for work that "casts a shadow bigger than the caster" — positioning that projects the capacity the project is growing into. He was explicit that this is different from dishonesty: "not because they lie, but because they know they will grow to fill those shoes." Hold both halves of that sentence at once.

If a sentence you wrote would embarrass him in six months when he reads it again, rewrite it.
If a sentence you wrote is true today *and* true of where the project is heading, keep it.

That is the line.
