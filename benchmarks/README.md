# semantix-ai benchmarks

Reproducible benchmarks comparing semantix's local NLI judge against LLM-judge alternatives across integrations.

## Layout

- `common/` — judge adapters, metrics, runner, IO, cache
- `dspy/` — DSPy integration benchmarks
  - `customer_support/` — 200-example custom task
  - `hotpotqa_groundedness/` — 200-example HotpotQA subset

## Running

1. `cp benchmarks/.env.example .env` and fill in `GROQ_API_KEY` and `GEMINI_API_KEY` at repo root.
2. `pip install -r benchmarks/requirements.txt -e .`
3. `python -m benchmarks.dspy.customer_support.run` or `python -m benchmarks.dspy.hotpotqa_groundedness.run`

## Results

Each run writes to `results/raw.csv`, `results/summary.md`, and `results/run_metadata.json`. Notebooks under each task render charts and narrative.

## Free-tier execution

All runs use free-tier APIs only:

- **Groq** — Llama 3.3 70B at 6000 RPD
- **Gemini 2.5 Flash** — 250 RPD (operational proxy-ground-truth on full 200 examples)
- **Gemini 2.5 Pro** — 25 RPD (verification slice, 25 examples)

Judgments are cached in `benchmarks/.cache.sqlite` keyed on `(judge, text, intent)`. Errors are NOT cached, so a run that hits a daily quota can be re-run the next day and will resume from where it stopped. Expect ~3–5 wall-clock days per task on free-tier quotas.
