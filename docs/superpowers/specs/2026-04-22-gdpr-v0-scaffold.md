# GDPR-v0 Sibling Model — Scaffold Spec

> **Status:** scaffold shipped 2026-04-22. Fine-tuned weights pending — see v0.3 roadmap.

## Goal

Ship a GDPR-fine-tuned sibling to `labrat-aiko/nli-popia-v1` using the same
recipe (hand-authored seeds → LLM paraphrase → fine-tune → ONNX quantise →
hash-pinned eval → release-gate). The scaffold in this commit gives us the
stable public API (`from semantix import GDPRJudge`, `from semantix.presets.gdpr
import GDPR_CONSENT, ...`) without blocking on the fine-tune itself.

This is the first instantiation of the "sibling-model velocity moat"
recommendation — the hypothesis is that shipping a second regulation using
the same recipe in under three weeks is a stronger positioning signal than
a single well-polished POPIA release.

## Why GDPR next (and not HIPAA)

- **Market pull.** GDPR has the largest installed base of privacy-aware
  engineering teams globally. Every EU-operating SaaS has a compliance
  surface that could consume this judge.
- **Recipe transferability.** GDPR and POPIA share the same conceptual
  skeleton (consent, minimality, security, breach, transfers, rights).
  Eight of the 21 seeds are near-direct counterparts of the POPIA seeds
  already authored.
- **Reuse of the base NLI model.** Same `cross-encoder/nli-MiniLM2-L6-H768`
  backbone, same tokenisation, same quantisation variants. Net-new work
  is limited to seed authoring, eval authoring, and the fine-tune run.
- **HIPAA is deferred.** HIPAA needs domain SMEs (covered entity vs.
  business associate, PHI definitions, breach thresholds) that neither the
  author nor an LLM can write accurately from public knowledge alone. That
  is v0.4 work, once GDPR-v1 is shipped and a clinical reviewer is engaged.

## What this commit ships

1. **21 hand-authored seeds** in `data/gdpr_seeds.jsonl`, 3 per clause ×
   7 canonical clauses:
   - Consent (Art. 6(1)(a), Art. 7)
   - Minimality (Art. 5(1)(c))
   - Security (Art. 32)
   - Breach notification (Art. 33, Art. 34)
   - Cross-border transfers (Chapter V)
   - General lawful processing (Art. 5(1)(a))
   - Data subject rights (Art. 15–22)

   Label distribution: 8 contradiction, 7 entailment, 6 neutral. All EU
   context (CNIL, Irish DPC, SCCs, AWS Frankfurt, LinkedIn scraping, etc.).

2. **Seven Intent presets** in `semantix/presets/gdpr.py`:
   `GDPR_CONSENT`, `GDPR_MINIMALITY`, `GDPR_SECURITY` (threshold 0.85),
   `GDPR_BREACH` (negate=True), `GDPR_TRANSFERS`, `GDPR_PROCESSING`,
   `GDPR_DATA_SUBJECT_RIGHTS`. Each references specific GDPR articles in
   its description string.

3. **`GDPRJudge`** in `semantix/judges/gdpr.py`. Subclass of
   `QuantizedNLIJudge` pointing at `labrat-aiko/nli-gdpr-v1`. Wraps
   `_load_session` and `_load_tokenizer` in `try/except` so that until the
   real weights are published, the judge transparently falls back to
   `labrat-aiko/nli-popia-v1` and emits a `RuntimeWarning` explaining the
   degraded performance. Top-level re-exported from `semantix` with the
   same optional-import pattern as `POPIAJudge`.

4. **Training scripts** in `scripts/`:
   - `expand_gdpr_seeds.py` — parameterised clone of the POPIA expander,
     with an EU-member-state prompt (supervisory authorities, provinces).
   - `train_gdpr.py` — parameterised clone of the POPIA trainer, writing
     to `out/nli-gdpr-v1` and reading a hash-pinned
     `scripts/_gdpr_eval_hash.txt` (to be created when the eval set is
     authored).

5. **`[gdpr]` extra** in `pyproject.toml` mirroring `[popia]`.

## What is *not* shipped in this commit

- `data/gdpr_eval.jsonl` (target 150 hash-pinned holdout pairs)
- `data/gdpr_train.jsonl` (the expanded ~600-row training set)
- `scripts/_gdpr_eval_hash.txt`
- Actual fine-tuned weights on Hugging Face
- A GitHub Actions release-gate that blocks regression on GDPR macro F1
- A GDPR demo Space

## v0.3 roadmap (order of operations)

1. **Open a seed-contribution issue** (`gdpr-v0-seeds`) on the public
   repo. Invite EU privacy engineers to add premise/hypothesis pairs in
   the existing JSONL schema. Target 60+ hand-authored seeds before
   expansion.

2. **Author the eval set.** 150 pairs, hash-pinned in
   `scripts/_gdpr_eval_hash.txt`, released as a dataset
   (`labrat-aiko/gdpr-compliance-nli`) alongside the model.

3. **Run `scripts/expand_gdpr_seeds.py`** to produce ~600 training rows
   via LLM paraphrase (same tooling as POPIA).

4. **Run `scripts/train_gdpr.py`** (GPU, ~3 epochs, ~30 min on a single
   A10G). Upload quantised variants to
   `labrat-aiko/nli-gdpr-v1`.

5. **Release semantix-ai v0.3.0** — bump `_REPO_ID` usage path (no code
   change required; the runtime fallback evaporates once the repo
   resolves). Add GDPR macro F1 to the release-gate.

6. **Demo Space.** Clone the POPIA Space, swap the model and clauses,
   publish as `labrat-aiko/gdpr-judge-demo`.

## API stability contract

The public surface shipped in this commit is intended to be stable
across the v0.3 fine-tune:

- `from semantix import GDPRJudge` — instantiation API unchanged
- `from semantix.presets.gdpr import GDPR_CONSENT, ...` — seven names,
  stable
- `GDPRJudge.clauses()` classmethod — canonical seven-entry list
- `GDPRJudge.recommended_threshold` — `0.70` (may be retuned in v0.3
  based on the eval macro F1 optimum)

If v0.3 retunes `recommended_threshold` the change will be called out in
`CHANGELOG.md` with the measured impact on the hash-pinned eval.

## Risk and failure modes

- **Fallback confusion.** A user who installs `[gdpr]` and calls
  `GDPRJudge()` today gets POPIA weights with a warning. If they filter
  warnings or ignore stderr, they may report "GDPR scores look like
  POPIA scores" — because they literally are. **Mitigation:** the
  `RuntimeWarning` is emitted with `stacklevel=2` so it surfaces at the
  call site, and the judge sets `self._repo_id` to the actually-loaded
  repo so diagnostic code can distinguish. The README and model-card
  stub will both say "v0 = POPIA weights, v1 = fine-tuned".
- **Regulator overreach.** Seven single-sentence hypotheses are a
  tractable but not legally-complete reading of GDPR. The same
  intended-use framing from the POPIA model card must ship verbatim on
  the GDPR card: this is a semantic-consistency tool, not a compliance
  determination.
- **Seed quality.** The 21 seeds are the author's best-effort
  engineering reading, not legal advice. The community-contribution
  issue is the primary lever for driving quality before the fine-tune.

## References

- POPIA fine-tune spec: `docs/superpowers/specs/2026-04-21-popia-finetune-design.md`
- POPIA model card: <https://huggingface.co/labrat-aiko/nli-popia-v1>
- POPIA dataset: <https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli>
- SA Information Regulator outreach: `docs/outreach/info-regulator-letter.md`
