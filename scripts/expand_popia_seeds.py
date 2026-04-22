"""Expand POPIA seed NLI pairs into a training set via LLM paraphrase.

Reads data/popia_seeds.jsonl, calls an LLM to produce ~30 variants per seed
preserving the label, deduplicates, and writes data/popia_train.jsonl.

Usage:
    OPENAI_API_KEY=... python scripts/expand_popia_seeds.py
    # or
    ANTHROPIC_API_KEY=... python scripts/expand_popia_seeds.py --provider anthropic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SEEDS_PATH = Path("data/popia_seeds.jsonl")
TRAIN_PATH = Path("data/popia_train.jsonl")
VARIANTS_PER_SEED = 30

EXPAND_PROMPT = """You are expanding a natural-language-inference training example.

Given one (premise, hypothesis, label) triple, produce {n} NEW variants that
preserve the label exactly. Each variant MUST:
- vary business domain (SaaS, banking, healthcare, retail, government, NGO)
- vary company size (startup, SME, enterprise)
- vary SA province context where plausible (Western Cape, Gauteng, KZN, etc.)
- vary phrasing register (casual, formal, legalese)
- keep the NLI label identical to the original

The clause (POPIA concept) is: {clause}
The original label is: {label}

Original premise: {premise}
Original hypothesis: {hypothesis}

Output ONLY a JSON array of {n} objects, each with keys "premise" and
"hypothesis". No commentary, no markdown fencing, no extra fields.
"""


def call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def expand_seed(seed: dict, provider: str) -> list[dict]:
    prompt = EXPAND_PROMPT.format(
        n=VARIANTS_PER_SEED,
        clause=seed["clause"],
        label=seed["label"],
        premise=seed["premise"],
        hypothesis=seed["hypothesis"],
    )
    raw = call_anthropic(prompt) if provider == "anthropic" else call_openai(prompt)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    rows = []
    for v in data:
        if not isinstance(v, dict) or "premise" not in v or "hypothesis" not in v:
            continue
        rows.append(
            {
                "clause": seed["clause"],
                "premise": v["premise"],
                "hypothesis": v["hypothesis"],
                "label": seed["label"],
                "scenario": seed.get("scenario", "synthetic"),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = ap.parse_args()

    if not SEEDS_PATH.exists():
        print(f"missing {SEEDS_PATH}", file=sys.stderr)
        return 1

    seeds = [json.loads(line) for line in SEEDS_PATH.read_text().splitlines() if line.strip()]
    print(f"expanding {len(seeds)} seeds -> target ~{len(seeds) * VARIANTS_PER_SEED} rows")

    seen: set[str] = set()
    out: list[dict] = []
    for s in seeds:
        key = hashlib.sha256(f"{s['premise']}|{s['hypothesis']}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(s)

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] expanding {seed['clause']}/{seed['label']}...", flush=True)
        try:
            variants = expand_seed(seed, args.provider)
        except Exception as e:
            print(f"  failed: {e}; skipping", file=sys.stderr)
            continue
        kept = 0
        for v in variants:
            key = hashlib.sha256(f"{v['premise']}|{v['hypothesis']}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
            kept += 1
        print(f"  kept {kept}/{len(variants)} (running total: {len(out)})")
        time.sleep(0.5)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(out)} rows to {TRAIN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
