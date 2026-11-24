# Development Roadmap — Now / Next / Later

> **Status:** approved 2026-09-15. Reviewed at every release: finished items move, decisions get
> logged below, and Later triggers are re-checked.

## Goal

Make the POPIA stack trustworthy end to end — judge, presets, and audit trail — before widening
scope. Every public claim about a score, a gate result, or a preset must hold on the model files
users actually load.

## Why: what verification found on 2026-09-15

`evaluate_popia` on `data/popia_eval.jsonl` (SHA-256 matches `scripts/_popia_eval_hash.txt`),
POPIA v1 vs stock, each ONNX file forced with `model_variant=`. One machine (i5-11320H, which has
AVX-512 VNNI) unless noted.

| File | Runtime | Stock F1 | POPIA F1 | Release gate |
|---|---|---|---|---|
| `model_quint8_avx2` | Windows, Python 3.14, onnxruntime 1.24.4 | 0.5512 | 0.8022 | **FAIL**: minimality 0.708 < stock 0.738 |
| `model_quint8_avx2` | Linux, Python 3.12, onnxruntime 1.18.1 | 0.5512 | 0.7920 | **FAIL**: minimality |
| `model_qint8_avx512` / `_avx512_vnni` / `_arm64` (identical output) | Windows, onnxruntime 1.24.4 | 0.5170 | 0.8217 | PASS |
| `model_qint8_avx512_vnni` | Linux, onnxruntime 1.18.1 | 0.5170 | 0.8134 | PASS (the preprint's 0.517 / 0.813) |
| CI run 34934990433 (auto-selected file; runner CPU not logged) | Linux, pip-resolved | 0.4991 | 0.8115 | PASS |

Other verified findings:

- File selection reads `/proc/cpuinfo`, so Windows and Intel macOS always load the AVX2 file,
  even on AVX-512 CPUs. Linux on the same CPU loads VNNI.
- v1's files came from `scripts/train_popia.py`, which has no artifact gate. The per-file
  artifact gate in `scripts/train_popia_v2.py` (added 2026-07-15, after v1 and v2 were uploaded)
  rejects collapse and a >0.10 drop vs PyTorch, not the release gate's per-clause rule.
- All three POPIA preset integration tests fail on every file at the 0.75 threshold: consent
  0.116–0.256, cross-border 0.001–0.003, breach 0.001–0.002. Preset descriptions are compound
  requirements; the holdout's hypotheses are short, single-claim, and scenario-specific.
- With the stock judge (AVX2 file, Windows, onnxruntime 1.24.4), "polite and professional
  customer service" scores 0.072 on a polite, thankful reply; "The text expresses gratitude."
  scores 0.953.
- `GDPRJudge().evaluate()` raises `AttributeError`: its `__init__` never sets `_temperature`.
- Only the decorator splits composites into leaves. `assert_semantic`, the five integrations, and
  `StreamCollector` send the concatenated docstring in one judge call. `integrations/guardrails.py`
  and `StreamCollector` resolve thresholds differently from everything else.
- Judges download the latest `main` of each model repo; no revision is pinned.
- `AuditEngine.flush()` truncates the file before rewriting it.
- The preprint says the v2 holdout has 48 pairs; the file has 47, and the paper's own n = 197
  union only adds up with 47.
- POPIA-Bench `RESULTS.md` has blank reference scores while its README lists ~0.48 / ~0.78 / ~0.78.
- Context: PyPI is on 0.3.2 with master ahead (PR #3 unreleased). Hugging Face 30-day downloads:
  `nli-popia-v1` 28, `nli-popia-v2` 0, `sa-compliance-embeddings-v1` 75, `popia-instruct-v0` 8,
  dataset `popia-compliance-nli` 37. GitHub: 0 issues ever, 0 open PRs.

## Principles

- **Claims ship only with evidence.** Presets stay marked unvalidated until they pass their gate.
  Every quoted F1 or gate result names its model file and runtime.
- **Two lanes.** Engineering never waits on labels; claims never ship ahead of labels.
- **One item, one design.** Each item gets its own spec and plan when it starts. 0.3.3 is small
  enough to go straight to implementation.
- **Scope.** Product and codebase only; distribution and outreach are Akhona's lane.

## Now (~3–4 weeks)

1. **0.3.3 (this week)**
   - Release master's unreleased fixes from PR #3 (the CHANGELOG entry is written).
   - Fix the `GDPRJudge` crash and add its first test.
   - Align the three flat "~15 ms" claims (`README.md:211`, `README.md:295`,
     `docs/integrations/dspy.md:108`).
   - Copy the "unvalidated" caveat from `docs/judges.md:163` into the `presets/popia.py` docstring.
   - Correct the determinism wording to name what moves scores (model file, CPU, onnxruntime
     build) and state v1's per-file gate results, including the AVX2 failure.
   - Confirm with a clean-venv `pip install semantix-ai==0.3.3`.
2. **Release gate over every shipped file**
   - `evaluate_popia` runs each file via `model_variant=`, with the threshold passed explicitly as
     0.5 so a later default change can't move the gate.
   - The report records file, onnxruntime version, OS, and CPU flags.
   - `popia-eval.yml` requires every file to pass; the artifact gate in `train_popia_v2.py` adds
     the no-per-clause-regression rule.
   - This intentionally fails on v1's AVX2 file until item 3 resolves it.
3. **AVX2 experiment (throwaway; the output is a decision)**
   - On an AVX2-only machine (a GitHub runner if `lscpu` confirms no AVX-512), compare scores and
     latency of the AVX2 file against a qint8 file.
   - Decide between (a) one qint8 file everywhere — the three qint8 files already agree exactly;
     (b) re-quantize the AVX2 file and re-gate it; (c) keep it and document it.
   - If per-CPU files stay, fix CPU detection on Windows and macOS.
4. **0.4.0: one evaluation path**
   - One internal module owns the default-judge fallback, threshold precedence, and composite
     decomposition. The decorator, `assert_semantic`, `StreamCollector`, the five integrations,
     and the CLI all route through it.
   - A parametrised test asserts every entry point returns the same verdict for the same text,
     intent, and judge, composites included.
   - Minor bump because verdicts change: composites start working in the integrations, and the
     Guardrails and streaming thresholds change.
5. **Preset labels setup**
   - Scaffold `data/popia_preset_eval.jsonl` with 70 slots (10 per preset) and its validator and
     `--pin`, so writing can start in week 1.

**Done when:** 0.3.3 and 0.4.0 are live and confirmed; every shipped v1 file passes the gate or is
publicly marked as failing; the AVX2 decision is logged below; the preset file is committed.

## Next — engineering lane (~6–8 weeks, no new labels)

Order: identity, then verdicts, then audit, so certificates can record what produced each score.

**0.5 — Reproducible judges**
- Pin model revisions (v1 `3074d1991660f137518f5e6e071ad75b88ea40fb`, v2
  `f602d3e136088ff6bb6e83671da83cd2f6141d74`), with a `revision=` override for upgrades.
- `judge.identity`: repo, revision, ONNX file, onnxruntime version, OS. Used by the gate report
  and by certificates.
- Ship the AVX2 decision from Now.

**0.6 — Three-way verdicts and strict mode**
- `Verdict` gains the three NLI probabilities and a label (entailment / neutral / contradiction),
  `None` for non-NLI judges. Carried through all 8 construction sites (including `ForensicJudge`
  and composite combination) and onto `SemanticIntentError`.
- `on_uncertain` policy for negated intents, defaulting to today's behaviour (neutral passes).
- `strict=True` makes neutral fail negated intents and makes unresolvable annotations raise.
- `evaluate()` defaults to `recommended_threshold` (TrustMesh #4); safe once the gate pins 0.5.
- Docs for the open TrustMesh items: async usage, the in-memory audit model,
  `recommended_threshold` as an attribute, `POPIAJudge` in the README.

**0.7 — Audit sink**
- `@validate_intent(audit=True | engine)` records certificates from verdicts the decorator
  already computes, including `judge.identity` and the three-way label.
- `AuditEngine(path=..., autoflush=True)` appends and fsyncs each record; `flush()` becomes
  atomic (write a temp file, then rename).
- `checkpoint()` returns (last hash, count) to store outside the chain — the prerequisite for
  detecting truncation or rewrites.

## Next — evidence lane (at the pace of labelling)

**E1. Preset label set** — `data/popia_preset_eval.jsonl`
- 7 presets × 10 premises = 70, each labelled with the outcome the preset check should reach:
  satisfies / violates / not enough evidence, split 4 / 4 / 2 per preset.
- Premises mix memo, news, and first-person styles.
- The validator refuses to pin on a duplicate of any training input the training scripts read
  (seeds, paraphrases, `popia_train.jsonl`), either holdout, or POPIA-Bench, or on a wrong split.
- Akhona writes; Claude builds the validator and trap checks. The hash is pinned in a standalone
  commit.

**E2. Preset rewrite** (iterates against E1 without relabelling)
- Rewrite each preset as short single claims in the holdout's style; compound presets become
  composites of single claims.
- Make presets usable with `@validate_intent` while keeping today's instance form working.
- Tune each preset's threshold on the 150-pair holdout, never on E1.

**E3. Preset gate** — a preset is validated only if, on every shipped model file, no *violates*
premise passes, no *satisfies* premise is blocked, and no *not enough evidence* premise passes as
satisfied. One error fails the preset. Failing presets stay marked unvalidated in code and docs;
GDPR presets stay unvalidated until a GDPR set exists.

**E4. Paper and model cards** (submission and live Hugging Face edits stay Akhona's call)
- Per-file, per-runtime results table and a deployment-variance limitation in the preprint; run
  v2 on every file against both holdouts.
- Correct "48 pairs" to 47.
- Fill POPIA-Bench reference scores per file with `score.py` once 0.6 exposes three-way labels,
  and make `RESULTS.md` and the bench README agree.
- Bring v1's model card into the repo at `out/nli-popia-v1/README.md` (force-added, since `out/`
  is gitignored) and correct its numbers.
- A presets section in the paper if E3 passes, a limitation if it doesn't.

**Done when:** every preset is validated or explicitly marked unvalidated, and every number in the
paper names its file and runtime.

## Later (each starts only when its trigger fires)

| Item | Trigger | Why it waits |
|---|---|---|
| GDPR judge: finish the 54 remaining pairs, `--pin`, train, gate on every file | Preset labels pinned, and Akhona chooses to resume | `train_gdpr.py` won't run without the pin; authoring competes for the same hours |
| Synthetic vs hand-written data experiment | Akhona decides on sending seeds to an outside LLM, or picks a local model | Scripts are written but untracked |
| v2 as default, plus presets for its 3 extra clauses | v2 passes the all-files gate on both holdouts and its presets validate | Coverage isn't accuracy: the preprint reports v2 at 0.747 vs v1's 0.813 on v1's clauses (file and runtime unstated) |
| External anchoring or signed certificates | 0.7's `checkpoint()` has shipped and someone needs third-party-verifiable audit | Signing needs key management; nobody has asked |
| DSPy benchmark rerun | Before any benchmark number is reused publicly, or after 0.5 changes model loading | Results come from 0.2.0 (run 2026-04-22) |
| Embeddings in the library (e.g. clause retrieval) | Downloads hold up and a concrete library use appears | Not part of the judge story yet |
| Pro tier (`2026-04-13-pro-tier-design.md`) | A confirmed external production user asks for hosted features | No evidence of demand |
| Compliance proxy (first step of the parked IDE idea) | The triggers Akhona set in May 2026 | Parked |

## Not on this roadmap

- Retrying v3 or larger base models under INT8 (closed 2026-07-15).
- New jurisdictions beyond GDPR.
- Competing on generic-validator features.
- Distribution and outreach.

## Open questions (resolved in the relevant spec)

- ~~AVX2 path: (a), (b), or (c)~~ — **decided 2026-09-16: (a)**, one qint8 file everywhere.
  Evidence: `docs/experiments/2026-09-16-avx2-vs-qint8.md`. Implementation shape (a1 vs a2)
  is still open; see that report.
- Audit sink: one certificate per attempt or per call? Leaning per attempt, linked by a call id,
  since rejected outputs are evidence the guard worked.
- Audit sink: singleton or injected engine? Leaning both, with `audit=True` meaning the singleton.
- Preset API: how presets become usable with the decorator without breaking
  `preset.description` for existing callers.

## Decision log

- 2026-09-16: **AVX2 path decided — option (a), one qint8 file everywhere.** Measured on four
  AVX2-only GitHub runners plus a VNNI control (`docs/experiments/2026-09-16-avx2-vs-qint8.md`):
  `model_quint8_avx2.onnx` is worse on both accuracy and latency on both CPU classes (AVX2-only
  +0.0277 F1 and ~20% faster for qint8; VNNI +0.0452 F1 and 2.8× faster), and it fails the
  per-clause rule on minimality everywhere. The file stays in the model repo because installed
  versions <= 0.3.3 request it by name, so this does not by itself turn `popia-eval` green — the
  a1/a2 choice does. Also recorded: the stock qint8 baseline collapses without VNNI
  (0.4991 -> 0.2462) while POPIA holds (0.8115 -> 0.8004), so the gate's delta must not be quoted
  as fine-tune evidence on AVX2-only hardware.
- 2026-09-16: v0.3.3 shipped Now items 1 and 2 (all-files gate, GDPRJudge fix, wording).
- 2026-09-15: Horizon is Now / Next / Later. Next is built around a trustworthy POPIA stack.
  Akhona writes the preset labels; the set is 10 per preset (70). The quarter runs as two lanes
  with claims gated on evidence. 0.3.3 discloses v1's AVX2 gate failure before the fix is chosen.
  GDPR authoring waits until the preset labels are pinned.
