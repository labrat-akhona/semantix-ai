# POPIA-Bench v1 — Results

Public leaderboard for the [POPIA-Bench v1](README.md) benchmark. Submissions are accepted via pull request to this file. See the [README](README.md#how-to-submit) for submission requirements.

## Reference scores (computed at training time)

| Model | Macro F1 | Note |
|---|---|---|
| `labrat-aiko/nli-popia-v2` | _to be filled in once full bench is scored_ | The model whose release-gate uses this bench |
| `labrat-aiko/nli-popia-v1` | _N/A_ | v1 covered only the first 7 clauses; included for historical comparison |
| `cross-encoder/nli-MiniLM2-L6-H768` (stock) | _to be filled in_ | Baseline before any POPIA fine-tuning |

## Community submissions

_Empty as of v1 release. Add your entry below via pull request._

| Date | Model | Macro F1 | Per-clause F1 table | Submitter |
|---|---|---|---|---|
| | | | | |

## Reproducibility

Every entry on this leaderboard should be reproducible from the cited model's public artifacts. We do not require weights to be released, but we do require a recipe that lets a third party reach within $\pm 0.01$ of the reported macro F1.
