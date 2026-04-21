# DSPy Benchmark & Merge Campaign — Design Spec

**Date:** 2026-04-21
**Author:** Akhona Eland (with Claude)
**Status:** Approved, pending implementation plan

## 1. Problem

`stanfordnlp/dspy` PR #9583 (adding semantix-ai to their providers/integrations table) was closed on 2026-04-10 by `@isaacbmiller` with the feedback:

> "For our providers with integrations, we try to keep it specific to people who have built, measured, and tested specific DSPy features. Are there any DSPy-specific features in Semantix? Or things that get better for your users via an integration?"

Semantix does have DSPy-specific integration code already shipped: `semantix/integrations/dspy.py` (155 lines, `semantic_reward` + `semantic_metric` targeting `dspy.Refine` / `dspy.BestOfN` / `dspy.Evaluate` / `MIPROv2`). The PR bounced because it was docs-only and failed to surface or quantify that work.

Three other "official monorepo" closures from the same outreach wave (pydantic-ai, langchain, instructor) share a different root cause — those projects have policy-level refusals to accept integrations in-tree. DSPy is the exception: the maintainer asked for **evidence**, not a venue change. That makes DSPy the highest-ROI merge target to re-attempt.

## 2. Goal

Produce evidence strong enough that a resubmitted DSPy PR is merged on methodological grounds. The evidence must show:

- semantix-ai `semantic_reward` is a **legitimate DSPy primitive** — i.e., it improves user outcomes, not just shifts them.
- It is **faster and cheaper** than LLM-judge rewards at comparable quality.
- The comparison holds on at least one task DSPy users recognize (not only a custom showcase).

Secondary goals that fall out for free:

- A reusable benchmark harness that cuts the effort of future integration pitches (outlines, marvin, llama_index) roughly in half.
- A Dev.to technical article that earns adoption independently of whether the PR merges.
- A patch release of `semantix-ai` (v0.1.13) that pins the benchmark artefacts to a version.

## 3. Non-goals (this cycle)

- No `semantix.integrations.outlines` module — deferred to next cycle.
- No human labels in the benchmark — proxy-ground-truth via a strong LLM is the rigor ceiling for v1.
- No custom NLI training — separate track.
- No changes to semantix's public API.
- No HuggingFace Datasets mirror of results — raw CSVs in GitHub are sufficient.

## 4. Approach

Approach 2 of three considered (Minimal / **Comprehensive** / Phased). "Comprehensive evidence pack" was selected because the stated bar ("merge-or-die") eliminates methodology holes a reviewer could reject on. The other two approaches either undershoot the rigor bar (Minimal) or risk the maintainer closing again before follow-up evidence lands (Phased).

Approach summary: two benchmark tasks × three judges × two experiments per task, run end-to-end on free-tier APIs, results committed to the repo and linked from a new DSPy PR and a Dev.to article.

## 5. Scope

### 5.1 Benchmark tasks

**Task A — `customer_support_qa` (custom, showcase).** DSPy program: `dspy.ChainOfThought("customer_message -> response")`. Dataset is 200 `(customer_message, intent)` pairs committed as `dspy/customer_support/dataset.json`, balanced across three intents (`polite`, `on_topic`, `declines_without_being_rude`) at roughly 67 examples per intent. Each example is scored once against its paired intent — the 200 figure is pairs, not 200 × 3.

**Task B — `hotpotqa_groundedness` (public, rigor).** DSPy program: `dspy.ChainOfThought("context, question -> answer")`. 200-example subset of HotpotQA, fixed-seed index list committed as `dspy/hotpotqa_groundedness/indices.json`. One intent: `answer is grounded in the provided context`.

### 5.2 Judges

- **`SemantixJudge`** — wraps `QuantizedNLIJudge` from semantix core. Subject under test.
- **`GroqJudge`** — Llama 3.3 70B via Groq free tier, used as the baseline LLM-judge (the stand-in for "the LLM-judge a DSPy user would realistically configure"). Verified working: ~380ms total round-trip, 20ms inference.
- **`GeminiFlashJudge`** — Gemini 2.5 Flash, used as the operational proxy-ground-truth for the full 200-example sweep. Verified working: ~900ms round-trip (includes thinking tokens).
- **`GeminiProJudge`** — Gemini 2.5 Pro, used on a 25-example verification slice to correlate with Flash rankings. Addresses the "why Flash and not Pro?" reviewer question proactively.

### 5.3 Experiments (per task)

- **Reward-agreement.** One generation per example. Three judges (semantix, Groq, Gemini Flash) score every output across the full 200-example sweep. A separate 25-example verification slice additionally runs Gemini Pro on the same inputs. Metrics: Cohen's κ and Pearson r between each judge and Gemini Flash on the full set; κ/r between Flash and Pro on the verification slice (to legitimize Flash as the operational proxy).
- **Optimization-impact.** `dspy.BestOfN(N=5)` run **twice** per task, once with semantix as `reward_fn` and once with Groq as `reward_fn`. Gemini Flash is *not* used as a reward here (that would be circular, since it's also the final judge). Final output for each example judged by Gemini Flash; report mean score, delta vs. baseline, and paired win-rate of semantix-selected outputs vs. Groq-selected outputs.

## 6. Architecture

### 6.1 File layout

```
benchmarks/
├── README.md
├── requirements.txt
├── .env.example
├── common/
│   ├── judges.py
│   ├── metrics.py
│   ├── runner.py
│   ├── io.py
│   └── cache.py
└── dspy/
    ├── customer_support/
    │   ├── task.py
    │   ├── dataset.json
    │   ├── run.py
    │   ├── notebook.ipynb
    │   └── results/ (raw.csv, summary.md, run_metadata.json)
    └── hotpotqa_groundedness/
        ├── task.py
        ├── indices.json
        ├── run.py
        ├── notebook.ipynb
        └── results/
```

### 6.2 Interfaces

```python
@dataclass
class JudgeResult:
    score: float          # 0.0–1.0
    latency_ms: float
    cost_usd: float       # 0.0 on free-tier; also stored as "paid_equivalent_usd"
    raw: str | None
    error: str | None

class Judge(Protocol):
    name: str
    def evaluate(self, text: str, intent: str) -> JudgeResult: ...
```

One protocol, four implementations. The runner is judge-agnostic: new providers plug in as another adapter in `common/judges.py`.

### 6.3 Data flow

```
task.py (examples)
   │
   ▼
DSPy program generates output(s)
   │
   ▼
all judges score each output (parallel where safe, serial on rate limits)
   │
   ▼
runner accumulates JudgeResult rows
   │
   ▼
io.write_csv + io.write_summary_md + run_metadata.json
   │
   ▼
notebook.ipynb loads raw.csv → charts & narrative
```

### 6.4 Reliability & reproducibility

- **Errors are row-level.** A failed judge call records `score=NaN` + `error` and continues. The run never aborts on a single failure.
- **Retries.** Exponential backoff once; on HTTP 429 read `Retry-After` and sleep. Max 3 attempts per call.
- **Determinism.** `temperature=0` on all LLM-judge calls. Committed datasets. DSPy global seed 42. `QuantizedNLIJudge` deterministic by construction.
- **Caching (opt-in).** `benchmarks/.cache.sqlite` (gitignored) keyed on `SHA-256(judge_name, text, intent)` → `JudgeResult`. Enabled with `--cache` flag. Critical for iterating on analysis without re-spending quota.
- **Metadata.** Each run writes `run_metadata.json` with timestamp, semantix version, model versions, git SHA.

## 7. Testing

- **Unit:** `semantix/tests/benchmarks/` — `test_judges.py` (HTTP mocks via `respx`), `test_runner.py` (stub judges), `test_metrics.py` (κ + r math).
- **Integration:** behind `--integration` flag; live 1-call smoke tests per provider. Not in CI.
- **CI:** GitHub Actions job triggered on `benchmarks/**` changes runs the unit test suite only. No live API in CI.

## 8. Costs

- **Plan cost:** $0. Groq free tier + Gemini free tier cover the full benchmark.
- **Call budget:**
  - Gemini Flash: 400 (agreement) + 800 (optimization final judging) = **~1,200 calls**, within ~5 days at 250 RPD.
  - Groq: 400 (agreement) + 2,000 (BestOfN reward scoring, 5 candidates × 200 examples × 2 tasks) = **~2,400 calls**, within one day at 6,000 RPD.
  - Gemini Pro: **25 calls** for the verification slice, within one day at 25 RPD.
  - Semantix: local, no quota.
- **Wall-clock:** Gemini Flash free tier is the binding constraint. ~5 days calendar if run serially, or overnight if paid tier is used later. Cache layer absorbs retries and mid-run tweaks.
- **Paid-equivalent disclosure.** Summary table reports both "free-tier actual" and "paid-tier equivalent" cost columns so readers see the comparison under real deployment economics.

## 9. Release & outreach sequencing

| # | Artifact | Destination |
|---|---|---|
| 1 | Benchmark infra + both tasks | `master` |
| 2 | Full benchmark run results | `master` |
| 3 | Rendered notebooks | `master` |
| 4 | `semantix-ai` v0.1.13 patch release | PyPI |
| 5 | Dev.to article | Dev.to |
| 6 | New DSPy PR (citing #9583, bringing evidence) | `stanfordnlp/dspy` |
| 7 | Social posts (LinkedIn, X, HN) | drafted in `articles/social-posts.md`, posted 48h after step 5 |

Steps 1–3 land before release so the PR body can link to materials already in `master`. Step 4 before 5 gives the article a "released today" CTA. Step 5 before 6 means the PR body cites published third-party-visible work. Step 7 last, so social posts can mention the open PR.

Estimated calendar: 5–7 days end-to-end.

## 10. Success criteria

**Primary (hard):** DSPy PR merged, or maintainer engages substantively with the evidence even if they ultimately decline. Either is a signal the methodology cleared their bar.

**Secondary (soft, worth tracking regardless of merge):**

- Dev.to article reaches ≥ 1,000 views in the first week.
- `semantix-ai` install count shows a detectable step-change within 14 days of publication.
- Benchmark harness is reused for at least one subsequent integration pitch within 30 days.

## 11. Open risks

- **Free-tier rate limits degrade or change** between design and run. Mitigation: cache layer means a partial run is resumable; fallback option to spend ~$5 on paid tier is preserved.
- **Groq returns non-numeric responses** intermittently. Mitigation: retry with a stricter reprompt; on second failure, record as error. Error rate reported in headline table (transparency beats suppression).
- **Benchmark shows semantix loses or ties** the LLM-judge on quality. Mitigation: still publish honestly. Speed / cost advantage alone is defensible for latency-sensitive use cases; losing gracefully is better credibility than not running the benchmark.
- **DSPy maintainer closes again** citing a different objection. Mitigation: the benchmark + article stand on their own merit regardless of DSPy's decision. No step in this plan is contingent on the merge.

## 12. Explicit decisions made during brainstorming

| Decision | Chosen | Alternative rejected |
|---|---|---|
| Benchmark scope | Comprehensive (2 tasks, 4 judges, 2 experiments) | Minimal viable proof / phased |
| Judge panel | semantix + Groq baseline + Gemini Flash proxy + Gemini Pro verification slice | Single baseline / paid Opus proxy / human labels |
| Cost | $0 via free tiers | $5–$18 paid tier |
| Release cadence | Single patch release v0.1.13 bundling everything | Rolling artifact releases |
| Outlines integration | Deferred to next cycle | Bundled with DSPy work |
| PR strategy | New PR citing the closed #9583 | Reopen #9583 |

## 13. Hand-off

Upon spec approval, invoke the `superpowers:writing-plans` skill to produce a step-by-step implementation plan, then execute via `superpowers:executing-plans`.
