# Changelog

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
