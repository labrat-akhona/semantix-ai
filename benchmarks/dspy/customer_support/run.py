"""Entry point: run agreement + optimization experiments for customer_support."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import dspy
from benchmarks.common.cache import JudgeCache
from benchmarks.common.io import write_csv, write_summary_md
from benchmarks.common.judges import GeminiFlashJudge, GeminiProJudge, GroqJudge, SemantixJudge
from benchmarks.common.runner import run_agreement, run_optimization
from benchmarks.dspy.customer_support.task import generate_all, load_examples, make_program

HERE = Path(__file__).parent
RESULTS = HERE / "results"

N_EXAMPLES = 50  # Full dataset is 200; 50 gives tight CIs on free-tier Flash quota
N_PRO_SLICE = int(
    os.environ.get("N_PRO_SLICE", "25")
)  # Set 0 to skip Pro slice when daily quota (25 RPD) is exhausted


def _dspy_lm_from_env() -> dspy.LM:
    """DSPy LM configured to use Groq as the generator (free tier)."""
    api_key = os.environ["GROQ_API_KEY"]
    return dspy.LM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )


def _cached(judge, cache: JudgeCache):
    """Wrap a judge so get/put hits the cache."""

    class Cached:
        name = judge.name

        def evaluate(self, text, intent):
            hit = cache.get(judge.name, text, intent)
            if hit is not None:
                return hit
            result = judge.evaluate(text, intent)
            cache.put(judge.name, text, intent, result)
            return result

    return Cached()


def main() -> None:
    load_dotenv()
    dspy.configure(lm=_dspy_lm_from_env())
    dspy.settings.rng = 42  # seed BestOfN

    RESULTS.mkdir(exist_ok=True)
    cache = JudgeCache(Path(__file__).parents[2] / ".cache.sqlite")

    examples = load_examples()[:N_EXAMPLES]
    print(f"[1/4] loaded {len(examples)} examples", flush=True)

    program = make_program()
    generated = generate_all(examples, program)
    print(f"[2/4] generated {len(generated)} responses", flush=True)

    semantix = _cached(SemantixJudge(), cache)
    groq = _cached(GroqJudge(), cache)
    flash = _cached(GeminiFlashJudge(), cache)
    pro = _cached(GeminiProJudge(), cache)

    agreement_rows = run_agreement(generated, [semantix, groq, flash])
    print(f"[3/4] agreement: {len(agreement_rows)} rows", flush=True)

    # Pro verification slice: first N_PRO_SLICE examples, Pro judge only
    slice_rows = run_agreement(generated[:N_PRO_SLICE], [pro])
    agreement_rows.extend(slice_rows)

    # Checkpoint: write agreement results immediately so we don't lose them if optimization crashes
    write_csv(agreement_rows, RESULTS / "raw.csv")
    print(f"[3.5/4] checkpointed {len(agreement_rows)} agreement rows", flush=True)

    def program_fn(input_dict, reward_fn):
        def adapted(_kwargs, pred):
            return reward_fn(pred.response)

        best = dspy.BestOfN(module=program, N=5, reward_fn=adapted, threshold=1.0)
        pred = best(**input_dict)
        return pred.response

    opt_rows: list = []

    def _on_example_done(ex_rows):
        opt_rows.extend(ex_rows)
        write_csv(agreement_rows + opt_rows, RESULTS / "raw.csv")

    try:
        run_optimization(
            generated,
            program_fn=program_fn,
            reward_judges=[semantix, groq],
            final_judge=flash,
            on_example_done=_on_example_done,
        )
        print(f"[4/4] optimization: {len(opt_rows)} rows", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[4/4] optimization FAILED at {len(opt_rows)} rows: {exc}", flush=True)

    rows = agreement_rows + opt_rows
    write_csv(rows, RESULTS / "raw.csv")
    write_summary_md(rows, RESULTS / "summary.md", task_name="customer_support_qa")

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_sha": git_sha,
                "examples": len(examples),
                "judges": [semantix.name, groq.name, flash.name, pro.name],
            },
            indent=2,
        )
    )

    cache.close()
    print(f"done -> {RESULTS}/")


if __name__ == "__main__":
    sys.exit(main())
