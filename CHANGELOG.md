# Changelog

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
