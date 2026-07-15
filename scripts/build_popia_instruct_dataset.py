"""Build the POPIA instruction dataset for popia-instruct-v0.

Everything is derived from sources we already have on disk:
  - data/sa_compliance_corpus.jsonl     (114 POPIA section texts from the official Act PDF)
  - data/popia_seeds*.jsonl             (labelled scenarios — entailment / contradiction / neutral)
  - data/popia_paraphrases*.jsonl       (paraphrased premises for the same clauses)

No LLM-generated content. Every instruction-response pair traces to a
section text or a labelled (premise, hypothesis, label) triple.

Output: data/popia_instruct.jsonl in chat-template format:
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

CORPUS = Path("data/sa_compliance_corpus.jsonl")
SEEDS = [Path("data/popia_seeds.jsonl"), Path("data/popia_seeds_v2.jsonl")]
PARAS = [Path("data/popia_paraphrases.jsonl"), Path("data/popia_paraphrases_v2.jsonl")]
OUT = Path("data/popia_instruct.jsonl")

SYSTEM = (
    "You are a careful South African data-protection assistant. Answer in clear, "
    "professional English with reference to the Protection of Personal Information "
    "Act, 2013 (POPIA). When citing sections, use the form 'POPIA §X'. Be specific, "
    "not vague. If a scenario is non-compliant, name the rule it breaks."
)

CLAUSE_TO_SECTIONS = {
    "POPIA consent": [11, 12, 18, 69],
    "POPIA minimality / purpose limitation": [10, 13, 14, 15],
    "POPIA security safeguards": [19, 20, 21, 22],
    "POPIA general processing": [8, 9, 4],
    "POPIA breach notification": [22],
    "POPIA cross-border transfers": [72],
    "POPIA data subject rights": [5, 23, 24, 25],
    "POPIA children's information": [34, 35],
    "POPIA special personal information": [26, 27, 28, 29, 30, 31, 32, 33],
    "POPIA automated decision-making": [71],
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    corpus = {int(r["section_id"].split("§")[1]): r for r in load_jsonl(CORPUS)}
    scenarios: list[dict] = []
    for p in SEEDS + PARAS:
        scenarios += load_jsonl(p)

    examples: list[dict] = []

    # ----- 1. Section explainers -----
    # "Explain POPIA §X" -> "POPIA §X is titled '<title>'. <text>"
    for num, doc in sorted(corpus.items()):
        # use the first ~600 chars of the section as the canonical "explanation"
        snippet = doc["text"]
        # canonical
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Explain POPIA §{num}."},
                    {"role": "assistant", "content": snippet},
                ]
            }
        )
        # title-only
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"What is the title of POPIA §{num}?"},
                    {"role": "assistant", "content": doc["title"]},
                ]
            }
        )
        # paraphrase question
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"What does Section {num} of the POPIA Act govern?",
                    },
                    {
                        "role": "assistant",
                        "content": f"POPIA §{num} is titled '{doc['title']}'. {snippet[:600]}",
                    },
                ]
            }
        )

    # ----- 2. Clause -> section listing -----
    for clause, sections in CLAUSE_TO_SECTIONS.items():
        sec_list = ", ".join(f"§{s}" for s in sections)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Which POPIA sections govern {clause.replace('POPIA ', '')}?",
                    },
                    {
                        "role": "assistant",
                        "content": f"The {clause.replace('POPIA ', '')} concern is anchored in POPIA {sec_list}.",
                    },
                ]
            }
        )

    # ----- 3. Scenario compliance review -----
    # For each labelled (premise, hypothesis, label), build a Q&A:
    #   user: "Review this scenario against [clause]: <premise>"
    #   assistant: "Compliant with..."  or  "Not compliant with..."  or  "Not directly governed by..."
    for s in scenarios:
        clause = s["clause"]
        clause_short = clause.replace("POPIA ", "")
        sections = CLAUSE_TO_SECTIONS.get(clause, [])
        sec_ref = " and ".join(f"POPIA §{n}" for n in sections[:2]) if sections else "POPIA"
        label = s["label"]
        if label == "entailment":
            response = (
                f"This scenario aligns with {sec_ref}. The processing described — '{s['premise'][:160]}' — "
                f"satisfies the requirement that '{s['hypothesis']}'."
            )
        elif label == "contradiction":
            response = (
                f"This scenario does NOT comply with {sec_ref}. The processing described — '{s['premise'][:160]}' — "
                f"violates the requirement that '{s['hypothesis']}'."
            )
        else:  # neutral
            response = (
                f"This scenario is not directly governed by {sec_ref}. While the activity described — "
                f"'{s['premise'][:160]}' — touches on data processing, it does not raise the specific "
                f"{clause_short} question this section was written to address."
            )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Review this scenario against {clause}: {s['premise']}",
                    },
                    {"role": "assistant", "content": response},
                ]
            }
        )

    # ----- 4. "What rule applies?" routing -----
    # Take each non-neutral scenario, ask the model to identify the relevant clause.
    for s in scenarios:
        if s["label"] == "neutral":
            continue
        clause = s["clause"]
        sections = CLAUSE_TO_SECTIONS.get(clause, [])
        sec_ref = " and ".join(f"POPIA §{n}" for n in sections[:2]) if sections else "POPIA"
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"What POPIA rule is implicated by this scenario? {s['premise']}",
                    },
                    {
                        "role": "assistant",
                        "content": f"This scenario implicates {sec_ref} — the {clause.replace('POPIA ', '')} provisions.",
                    },
                ]
            }
        )

    # Shuffle deterministically
    random.Random(42).shuffle(examples)

    with OUT.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {OUT} ({len(examples)} examples)")
    # Quick breakdown
    by_kind: dict[str, int] = defaultdict(int)
    for ex in examples:
        u = ex["messages"][1]["content"]
        if u.startswith("Explain POPIA"):
            by_kind["section explainer"] += 1
        elif u.startswith("What is the title"):
            by_kind["title"] += 1
        elif "Section" in u and "govern" in u:
            by_kind["paraphrase explainer"] += 1
        elif u.startswith("Which POPIA sections govern"):
            by_kind["clause -> sections"] += 1
        elif u.startswith("Review this scenario"):
            by_kind["scenario review"] += 1
        elif u.startswith("What POPIA rule"):
            by_kind["scenario routing"] += 1
        else:
            by_kind["other"] += 1
    print("by kind:", dict(by_kind))


if __name__ == "__main__":
    main()
