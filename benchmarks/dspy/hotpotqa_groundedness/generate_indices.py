"""One-shot: produce a 200-example deterministic subset of HotpotQA validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    random.seed(42)
    indices = random.sample(range(len(ds)), 200)
    out = Path(__file__).parent / "indices.json"
    out.write_text(json.dumps(sorted(indices)), encoding="utf-8")
    print(f"wrote {len(indices)} indices to {out}")


if __name__ == "__main__":
    main()
