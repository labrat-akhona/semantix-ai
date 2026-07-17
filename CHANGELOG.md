# Changelog

## v0.3.1 — Accurate audit-trail wording + adopter ergonomics (2026-07-17)

### Fixed
- **Removed an inaccurate "signed" claim from the README / PyPI description.** The
  audit certificate is **hash-chained**, not cryptographically *signed* — there is
  no keypair and no signature. Hash-chaining proves a chain is internally
  consistent (no entry was edited without redoing the rest); it does **not** prove
  authenticity or non-repudiation, since anyone who can rewrite the whole file can
  produce a chain that verifies. "signed" is a term of art with legal weight for a
  compliance reader, so the copy now says "a JSON-LD certificate hash-chained to the
  previous one … the math proves the chain is internally consistent." Actual
  cryptographic signing is a feature, not a docs change — it is not implied here.

### Added
- **`AuditEngine.reset()`** (finding #5) — a public classmethod to start a fresh
  chain, instead of poking the private singleton attributes. A second audit in the
  same process otherwise appends to the first one's chain.
- **Caller-supplied `id=` / `timestamp=` on `record()`** (finding #6) — both default
  to a fresh `uuid4` / wall-clock, but supplying them yields a reproducible,
  content-addressable certificate: identical inputs → byte-identical cert, so two
  runs of an audit can be diffed.
- **`QuantizedNLIJudge.calibrated` property + a warning** (finding #7) — reports
  whether a real temperature-scaling constant (`T != 1.0`) is actually in effect.
  The default v1 judge is uncalibrated, so its scores are systematically
  over-confident; `calibrated` makes that queryable, and requesting `calibrated=True`
  on a model with no `calibration.json` now raises a `UserWarning` instead of
  silently staying raw.
- **Actionable error when the optional inference deps are missing** (finding #9) —
  calling `QuantizedNLIJudge` / `POPIAJudge` without `onnxruntime` / `tokenizers`
  now raises a `ModuleNotFoundError` that names the fix (`pip install
  semantix-ai[popia]`) instead of a cryptic import error. The extra is also named in
  both judges' docstrings.

## v0.3.0 — First-class audit certificates + constancy detection (2026-07-16)

Found by dogfooding #2: the first production deployment of the audit certificate
(a POPIA §72 cross-border-transfer audit — 833 real transfers, 28 users, 3,332
hash-chained certificates). The chain verified, but the certificate could not say
*what* it certified. These changes make the differentiator do its job.

### Added
- **First-class certificate fields (`AuditEngine.record`).** New keyword-only
  parameters `hypothesis` (what was judged), `judge_id` (by which judge/version/
  config), `subject` (about whom), and `metadata` (any structured context). Before
  this, a caller had to JSON-encode all of it into the free-text `intent` string, so
  the "structured" audit trail was only as structured as each caller's ad-hoc JSON.
  All new params default to `None`, so code written against the 0.2.3 signature is
  unchanged.
- **`claim_hash`** — `sha256(premise, hypothesis, judge_id)`, added *alongside* the
  existing `output_hash`. Two certificates that judged *different clauses* against the
  *same premise* now have distinct `claim_hash` values (you can tell what was judged)
  while sharing an `output_hash` (premise-level constancy stays visible at a glance —
  that collapse is a real diagnostic: in the production audit 3,332 certs shared just
  4 `output_hash` values, making "the consent basis is identical for every user"
  cryptographically obvious). `output_hash` was **not** redefined.
- **`AuditEngine.chain_report()` / `summarize()` → `ChainReport`.** Integrity is not
  variety: a chain can verify perfectly while every certificate carries the same
  verdict. The report surfaces distinct verdict / claim / premise counts and an
  `is_constant` flag, so a constant chain is visible without a human having to notice.
  `semantix verify <trail.jsonl>` now prints these counts and a `!! CONSTANT CHAIN`
  warning. (Same one-line cure — count the distinct outputs — that the release
  artifact gate already applies to shipped models.)
- **`AuditEngine.verify_entries(entries)`** (static) and **`AuditEngine.load(path)`**
  — verify or resume a chain loaded from disk, including mixed v1/v2 chains.

### Changed
- **New certificates emit the `https://schema.semantix.ai/v2` `@context`.** The v2
  field set adds the items above. This is additive: the chain is verified by
  re-hashing each entry's whole JSON dict, so **existing `…/v1` certificates verify
  unchanged**, and a chain that upgrades mid-life (v1 rows then v2 rows) verifies as
  one intact chain. Pinned by a frozen v1 fixture regression test
  (`tests/fixtures/v1_chain_frozen.jsonl`, generated from 0.2.3 and never
  regenerated) so a future change that breaks v1 semantics fails loudly.
- **`QuantizedNLIJudge.evaluate` accepts `premise=` / `hypothesis=` aliases.** For an
  NLI/compliance use you call `evaluate(<premise>, <hypothesis>)`, but the published
  params are named `output` / `intent_description` (from the LLM-output-validation
  origin) and passing them backwards silently produced meaningless scores. The old
  names still work; supplying a legacy name *and* its alias for the same slot raises
  `TypeError` rather than silently picking one. The NLI mapping is now documented in
  the method docstring.

### Fixed
- **README oversell corrected (was: "records the judge identity and configuration").**
  Before v0.3.0 the certificate dict had neither field; now `judge_id` and `metadata`
  are first-class, so the docs are true rather than aspirational.

## v0.2.3 — Artifact gate + v2 tokenizer fix (2026-07-15)

### Fixed
- **`POPIAJudge(version="v2")` / `QuantizedNLIJudge` 404'd loading the
  tokenizer.** `_load_tokenizer` looked for `tokenizer.json` only at the
  repo root, but the HF repos store it in three different places —
  `nli-popia-v1` at the root, `nli-popia-v2` under `onnx/`, and
  `nli-popia-v3` under `pytorch/`. The loader now tries all three in turn.
  Anyone who `pip install`ed 0.2.2 and called `version="v2"` hit this;
  `v1` (root tokenizer, and the library default) was unaffected, which is
  why it stayed hidden. Verified live against all three repos.

### Added
- **Post-export artifact gate for quantized model releases.** The training
  pipeline now scores the *shipped ONNX file* after export/quantization —
  distinct-prediction count plus macro-F1 on both holdouts — and blocks the
  upload if the quantized artifact regressed. This closes a real failure
  mode: a model can pass a release gate that scored the *PyTorch weights*
  and still ship a dead *quantized* file. That is exactly how an earlier v3
  build shipped a constant predictor — it passed its own gate because the
  gate never scored the file users download. The gate now scores the
  artifact, and blocked a regressed build on its first outing.
- **`per_channel=True` INT8 quantization for DeBERTa bases** in the training
  script. Per-tensor dynamic INT8 collapses DeBERTa's disentangled
  attention into a constant predictor; RoBERTa/MiniLM bases (v1, v2) were
  unaffected and are unchanged.

### Notes
- **v3 (deberta-v3-base) was trained, evaluated, and shelved.** It beats v2
  at fp32 (0.785 vs 0.7465 macro-F1 on v1 clauses) but is *worse* than v2
  under the INT8 quantization the CPU-judge line requires (0.576), at ~3×
  the download. deberta-v3-base is a poor fit for a quantized-CPU judge, so
  **v2 (MiniLM base) remains the shipped generalist.** No public API change:
  the default judge is still `v1` and `_VERSION_TO_REPO` lists only v1/v2.

## v0.2.2 — Calibration (2026-05-18)

### Added
- **Opt-in probability calibration for `QuantizedNLIJudge` / `POPIAJudge`.**
  Passing `calibrated=True` fetches `calibration.json` from the model's HF
  repo and applies the fitted temperature constant at softmax so
  `verdict.score` is a well-calibrated probability (Guo et al., 2017).
  `nli-popia-v2` ships with `T*=2.5492`; ECE on a 116-pair stratified
  test split drops from 0.171 to 0.075 (−56.1%). Models without a
  `calibration.json` (base model, `nli-popia-v1`) silently fall back to
  `T=1.0`. Default remains `calibrated=False` for backwards compatibility;
  flipping the default is slated for v0.3.0 alongside a re-tuned
  `recommended_threshold`. Distinct from the existing *threshold*
  calibration in `semantix.training.calibrate` — see the calibration
  subsection of the POPIAJudge preprint (papers/popiajudge-arxiv/) for
  methodology and reliability diagrams.

### Fixed
- **Train-script reproducibility for `nli-popia-v2`.** Pin all RNGs
  (Python, NumPy, PyTorch, HF `set_seed`) via a new `--seed` flag, and
  build the dev split via per-(clause, label) stratification rather than
  tail-slicing the concatenated rows. The previous tail slice put the
  entire dev set in `v2_paraphrases`, so `load_best_model_at_end` had
  zero signal on v1 clauses and per-clause F1 could fluctuate ~20pp
  between runs. No effect on already-shipped `nli-popia-v2` weights.

## v0.2.1 — Dogfooding fixes (2026-05-15)

Two correctness bugs surfaced while integrating semantix-ai into a sibling
project ([TrustMesh](https://github.com/labrat-akhona)) during its Phase 2.5
integration. Triage notes for the full feedback set live at
`docs/backlog/2026-05-15-trustmesh-feedback.md`.

### Fixed
- **`@validate_intent` silently no-opped on unresolvable annotations.** If a
  decorated function's signature contained a forward reference or a
  `TYPE_CHECKING`-only import, `get_type_hints()` would raise; the decorator
  swallowed the exception and turned itself into a pass-through with no
  warning. `_resolve_intent_class` now logs a `WARNING` naming the function,
  the exception class, and the message before returning `None`. Silent
  no-op is the worst failure mode for a guardrail.
- **Composite intents scored near zero against NLI judges.** `@validate_intent(A & ~B & ~C)`
  fed the concatenated multi-clause description to the judge as a single
  premise/hypothesis pair, which cross-encoder NLI models can't entail —
  empirically ~0.02 on ideal output. The decorator now decomposes
  composites into leaves and calls the judge once per leaf, combining
  verdicts as `AllOf=all(passed)+min(scores)`, `AnyOf=any(passed)+max(scores)`,
  `Not=not(passed)+1-score`. Each leaf description is a single short
  sentence the model can directly entail or rule out.

### Notes
- Both bugs were found by dogfooding, not external adoption — semantix-ai
  has no confirmed external production users yet. The integration-driven
  bug-finding is the visible benefit of running one's own library against
  a sibling project rather than only against unit tests.

## v0.2.0 — The POPIA Release (2026-04-21)

### Added
- **POPIAJudge** (`semantix/judges/popia.py`) — NLI judge fine-tuned on South
  Africa's Protection of Personal Information Act. Loads the quantized ONNX
  model `labrat-aiko/nli-popia-v1` from HuggingFace Hub. 7 canonical POPIA
  clauses: consent, minimality, security safeguards, breach notification,
  cross-border transfers, general processing, data subject rights. Measured
  **macro F1 0.813** on a pinned 150-pair holdout (vs 0.517 stock), **+29.6pp**
  — every clause improved, no regressions.
- **POPIA presets** (`semantix/presets/popia.py`) — 7 pre-wired `Intent`
  instances, each pointing at `POPIAJudge` and the clause-specific threshold.
  Import as `from semantix.presets.popia import POPIA_CONSENT, ...`.
- **`semantix.eval` package** (`semantix/eval/popia.py`) — `evaluate_popia()`
  harness emits an `EvalReport` with per-clause F1, macro F1 delta, and a
  release gate (`delta >= 0.10` AND no per-clause regression).
- **`semantix eval popia` CLI subcommand** — Runs the release gate against
  the HF-bundled eval set and exits non-zero on failure. Used by CI to block
  regressions in the model artifact.
- **GitHub Actions release-gate workflow** (`.github/workflows/popia-gate.yml`)
  enforcing the 10pp macro-F1 delta on every push to `master`.
- **Training script** (`scripts/train_popia.py`) — Reproducible fine-tune
  from `cross-encoder/nli-MiniLM2-L6-H768` plus ONNX export with four CPU
  variants (AVX2, AVX512, AVX512-VNNI, ARM64). Requires `pip install
  'semantix-ai[train]'`.
- **`[popia]` and `[train]` extras** in `pyproject.toml` — Install the POPIA
  runtime with `pip install 'semantix-ai[popia]'` or training deps with
  `[train]`.
- **New CLI subcommands** (accumulated since v0.1.12):
  - `semantix prove` — re-runs the same validation N times and proves the
    judge is deterministic (bit-identical scores).
  - `semantix demo` — runs three canned scenarios (pass / fail / negated)
    in under a second; useful as a 60-second smoke demo.
  - `semantix verify <audit.jsonl>` — checks the hash chain on a tamper-
    evident audit trail file and prints a summary.
  - `semantix eval popia` — runs the POPIA release gate against the
    HF-bundled eval set and emits JSON when passed `--json`.

### Fixed
- **QuantizedNLIJudge label-order bug** — Since v0.1.5 the judge was reading
  `probs[2]` (neutral) as the entailment score instead of `probs[1]`. Scores
  now match the base model's `config.id2label` (`{0: contradiction, 1:
  entailment, 2: neutral}`). All four ONNX variants are bit-identical to
  before; only the Python-side label index changed.

### Notes
- POPIAJudge is the first compliance-specific model in the semantix
  ecosystem. The same fine-tune recipe (hand-authored seeds + paraphrases +
  release gate) is reusable for other regulatory frameworks (GDPR, HIPAA,
  EU AI Act clause libraries).

## v0.1.13 — 2026-04-21 (never published to PyPI; contents ship in v0.2.0)

### Added
- `benchmarks/` folder with a reproducible DSPy benchmark harness comparing semantix's local NLI judge against Groq Llama 3.3 70B, Gemini 2.5 Flash, and Gemini 2.5 Pro across two tasks (custom customer-support QA and a HotpotQA subset).
- Judge adapters, metrics (Cohen's kappa, Pearson r), SQLite cache, and runners live under `benchmarks/common/`.
- CI smoke test on `benchmarks/**` changes.

### Notes
- No public API changes in the `semantix` package itself.
- v0.1.13 was version-bumped internally but never uploaded to PyPI; install v0.2.0 to get this work.

## v0.1.5 — The Enterprise Performance Release (2026-04-06)

### Added
- **QuantizedNLIJudge** (`semantix/judges/quantized_nli.py`) — INT8 ONNX inference using `onnxruntime` + `tokenizers`. No PyTorch required. Auto-detects CPU architecture (AVX-512 VNNI / AVX-512 / AVX2 / ARM64) and downloads the optimal pre-quantized model from HuggingFace Hub.
- **ForensicJudge** (`semantix/judges/forensic.py`) — Wrapper that runs mask-perturbation saliency on failure to identify the top tokens driving contradiction. Injects a structured Breach Report into `Verdict.reason`.
- **AuditEngine** (`semantix/audit/engine.py`) — Thread-safe singleton that captures every validation as a JSON-LD Semantic Certificate. SHA-256 hash-linked entries create a tamper-evident audit trail.
- **`turbo` optional dependency** — Install with `pip install "semantix-ai[turbo]"` for zero-PyTorch inference.
- **Trust demo** (`tools/trust_demo.py`) — End-to-end demo of the Silent Guard, Detective, and Black Box working together on a legal review scenario.

### Changed
- **Turbo default** — `@validate_intent` now auto-selects `QuantizedNLIJudge` when `onnxruntime` is installed, falling back to `NLIJudge` otherwise.

## v0.1.4 — The Universal Standard Release (2026-04-06)

### Added
- **MCP Server** (`semantix/mcp/server.py`) — Exposes `verify_text_intent` as an MCP tool via FastMCP. Any MCP-capable agent (Claude Desktop, Claude Code, Cursor, etc.) can validate text against semantic intents. Returns JSON with score, passed, reason, and a structured `correction_suggestion` on failure for cross-agent self-healing.
- **MCP test suite** (`semantix/tests/test_mcp_server.py`) — 20 automated tests covering tool registration, response schema, correction suggestions, and graceful dependency error handling.
- **`mcp` optional dependency** in `pyproject.toml` — Install with `pip install "semantix-ai[mcp]"`.

### Fixed
- **NLI entailment index** — `NLIJudge` was reading the neutral logit (`scores[0][2]`) instead of the entailment logit (`scores[0][1]`). Label order is `{0: contradiction, 1: entailment, 2: neutral}`.
- **NLI softmax calibration** — Raw logits are now softmaxed via `apply_softmax=True`, producing true 0–1 probability scores instead of unbounded logits.
- **Hypothesis reframing** — `_to_hypothesis()` now converts imperative intent descriptions to progressive tense ("The text must politely decline..." → "Someone is politely declining..."), which scores significantly higher on NLI cross-encoders.

### Changed
- Updated README with MCP server section, Claude Desktop config snippet, and revised judge documentation.

## v0.1.3 — The Self-Healing Update (2026-04-05)

### Added
- **Informed Self-Healing** — `@validate_intent` injects structured `semantix_feedback` markdown into retries so the LLM knows what it got wrong (score, reason, requirement, rejected output).
- **Granular LLM Scoring** — `LLMJudge` returns 0.0–1.0 confidence + text reason instead of binary Yes/No.
- **Benchmarking** (`tools/benchmark.py`) — Proves self-healing improves reliability from 21.1% to 70.0%.
- **Self-healing test suite** (`semantix/tests/test_self_healing.py`).

### Changed
- Default judge switched from `LLMJudge` to `NLIJudge` (local, no API key required).

## v0.1.2 (2026-04-04)

- Fix install instructions, add NLI extra, improve metadata.

## v0.1.0 (2026-04-04)

- Initial release.
