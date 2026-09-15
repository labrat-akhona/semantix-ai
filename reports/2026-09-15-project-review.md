# Project review — 15 September 2026

Scope: installation, public Python API, real CLI use, audit persistence, tests,
documentation, and release tooling. Started against `c5c39bb` (v0.3.2).
Existing untracked experiments and drafts were preserved.

## Findings and implemented improvements

| Finding | Evidence | Change |
| --- | --- | --- |
| Documented positional decorators did not work | `@validate_intent(MyIntent, judge=...)` produced an Intent containing the function; calling it raised `TypeError` without judging text | Explicit Intent classes now select the contract, including composites, negation, and async retries; return annotations remain supported |
| Reset broke existing audit references | `engine = AuditEngine(); AuditEngine.reset(); engine.record(...)` raised a context-manager `TypeError` | Reset clears the shared chain while retaining a usable singleton and lock |
| Audit input crashed or replaced state with invalid data | JSONL rows such as `null`, arrays, nonnumeric scores, or array-valued intents reached summary logic; `load()` accepted broken chains | Shared UTF-8 reader validates consumed field types; errors identify the line; load validates links before replacing state |
| Invalid CLI options reached inference | NaN/infinite/out-of-range thresholds were accepted; nonpositive counts were silently coerced or sliced | Argument parsing rejects invalid values with exit code 2 |
| The live demo failed two advertised outcomes | Original gratitude-containing reply scored 0.146 for a vague politeness hypothesis; medication example scored 0.278 for a hypothesis requiring an absent dosage | Examples now explicitly test gratitude and recommending medication, with an opt-in real-model regression test and CI job |
| Setup and audit docs contradicted code | Base install has no inference dependencies; old guides claimed 25 MB quantization, automatic certificates, and stronger integrity guarantees | Runnable README example, accurate extras/model sizes, offline setup, v2 certificate schema, explicit recording and integrity limits |
| Public docs build included internal plans and failed strict validation | Historical plan linked to a nonexistent planned page | Working notes/plans remain in git but are excluded from the public site; strict docs build added to CI |

The full original integration suite was exercised, not only the mocked tests.
The three existing POPIA preset examples still fail their expected outcomes;
neither their assertions nor their model thresholds were weakened to pass.

## Verification evidence

- Before fixes: 36 new unit regression cases failed; the new live demo test also
  failed. Existing targeted cases passed.
- Final default suite: **438 tests passed**, with four integration tests
  deselected. This includes direct coverage for explicit intent precedence over
  a conflicting return annotation. The real demo integration test also passed.
- Real demo: the gratitude example passes, rude response fails, and medication
  recommendation is blocked by the negated check.
- README's complete first Python block ran with the real model, wrote an audit
  JSONL file in a temporary directory, and passed CLI chain verification.
- `semantix prove --n 20`: all 20 scores agreed to five decimal places; observed
  p50/p95/p99 inference latency was 27.7/43.5/49.3 ms on this machine. Repeatability
  measures score stability, not correctness; the compound default claim scored
  only 0.13233.
- Built wheel and source distribution; both passed Twine 7 validation. The older
  installed Twine 6.2 rejected metadata version 2.5 produced by current Hatchling;
  upgrading the temporary validation tool resolved this without package changes.
- Installed the wheel in a clean venv with no inference dependencies: core imports
  and `semantix --help` worked. Model inference requires an extra as documented.
- Strict MkDocs build and lint/format checks for tracked, configured source passed.
  Untracked research scripts have separate pre-existing lint findings and were
  not rewritten as part of these fixes.
- Independent code review ran 146 affected unit tests successfully and found an
  overload-ordering issue. The specific Intent overload was moved ahead of the
  generic callable overload.

Commands for reproduction (install `.[dev,docs]` for test/doc dependencies):

```bash
pytest -q
pytest tests/integration/ -m integration -v
semantix demo --no-color
semantix prove --n 20 --no-color
mkdocs build --strict
python -m build
python -m twine check dist/*
```

The review used Python 3.12, ONNX Runtime 1.26.0 and cached x86 AVX-512 VNNI
artifacts, with `HF_HUB_OFFLINE=1`. The isolated test invocation disabled unrelated
global pytest plugin autoload and explicitly loaded `pytest_asyncio.plugin`.

## POPIA results and limitations

The repository's 150-pair `data/popia_eval.jsonl` holdout matched the pinned SHA-256
`120e14a55bb653f4e7ce49e10c3618aaef21a064b63ca831def92aa244935461` and produced:

| Metric | Stock quantized NLI | POPIA v1 |
| --- | --- | --- |
| Accuracy | 0.7067 | 0.8333 |
| Binary macro-F1 | 0.5170 | 0.8134 |

F1 improvement was **0.2964**, and all seven per-clause scores improved over stock:
the existing release gate passed. This harness collapses contradiction and neutral
into non-entailment and uses the judges' `evaluate()` default threshold (0.5).
It does not validate the 0.75 threshold used by the preset integration helper.

The separate preset examples produced these entailment scores at the helper's
0.75 threshold:

| Existing scenario | Score | Expected outcome | Observed outcome |
| --- | --- | --- | --- |
| Agreement to privacy terms | 0.1158 | Consent passes | Fails |
| Frankfurt-to-Virginia replication | 0.0013 | Cross-border risk detected | Not detected |
| Notification in a quarterly newsletter | 0.0012 | Negated breach check blocks | Passes |

The short evidence and compound preset descriptions are not equivalent to the
holdout tasks. In particular, describing a foreign transfer does not establish
absence of a lawful basis, and a newsletter sentence supplies little incident
context. These are hypotheses explaining the mismatch, not independently verified
legal labels. They require representative examples and expert review. The published
holdout score alone does not justify deploying the presets as a compliance gate.

The local cache lacked the automatically selected v2 artifact and the CLI's hosted
eval file. Those offline calls correctly failed for unavailable files; the holdout
comparison above used the repository's local dataset. No training or model upload
was performed.

## Contextual research and next recommendations

1. **Prioritize preset evaluation at the actual deployment thresholds.** Separate
   short single-claim checks from compound requirements; evaluate representative
   positive, negative, and uncertain evidence with independently reviewed labels.
   Keep v1/v2 selection explicit: the v2 model card documents broader ten-clause
   coverage and a regression on the original seven-clause holdout. Expanding coverage
   is not automatically an accuracy upgrade. [Published v2 model card](https://huggingface.co/labrat-aiko/nli-popia-v2).

2. **Add an opt-in strict validation mode.** Unresolvable return annotations still
   warn and skip validation for compatibility. Negating non-entailment also treats
   uncertainty like absence. An explicit contract now avoids annotation-resolution
   failures; a future strict mode should reject missing contracts and distinguish
   uncertain evidence before applications use it for safety decisions. The base
   model predicts contradiction, entailment, and neutral, which supports evaluating
   a three-way policy. [Base model card](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768).

3. **Strengthen audit persistence before claiming independent authenticity.** Design
   an externally retained checkpoint (hash plus count), atomic file replacement,
   and an explicit decorator-to-audit sink. The current preceding-entry hashes
   cannot establish authorship or detect edited tails/full rewrites on their own.
   Certificate Transparency's signed tree heads and consistency proofs provide a
   useful design reference, not a claim that this library implements that protocol.
   [RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html).

4. **Keep model downloads and inference distinct in product copy.** Cached Hub
   downloads can still perform update requests. The refreshed setup documents
   `HF_HUB_OFFLINE=1` and its missing-cache behavior. [Hugging Face environment
   variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline).

CLI validation follows argparse's custom type conversion interface; development
configuration excludes nested worktrees while retaining existing repository rules.
[Python argparse](https://docs.python.org/3/library/argparse.html#type),
[Ruff configuration](https://docs.astral.sh/ruff/configuration/).

Historical research, outreach records, and old changelog entries are preserved as
dated records. Current user-facing documentation was refreshed; this review does
not claim that every historical metric or external model card was updated.
