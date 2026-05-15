"""Build the training corpus for sa-compliance-embeddings-v1.

Sources for v1:
1. POPIA Act text — split per section, used as the canonical retrieval target.
2. Existing labelled scenarios (data/popia_seeds*.jsonl, data/popia_paraphrases*.jsonl)
   — used as (scenario → clause) anchor/positive pairs.

Future v2 expansion: FSCA Nov 2025 AI report, SARB circulars, IR media statements.

Output: data/sa_compliance_pairs.jsonl  — JSONL of {"anchor": ..., "positive": ..., "source": ...}
        data/sa_compliance_corpus.jsonl — JSONL of {"section_id": ..., "title": ..., "text": ...}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pypdf

POPIA_PDF = Path("/home/aiko/.claude/projects/-mnt-c-Users-akhon-semantix/1dccf3a3-ab41-4bf3-a68e-5f46065d0f23/tool-results/webfetch-1778852283601-yy0cv2.pdf")
DATA_DIR = Path("data")
PAIRS_OUT = DATA_DIR / "sa_compliance_pairs.jsonl"
CORPUS_OUT = DATA_DIR / "sa_compliance_corpus.jsonl"

# Map POPIA-Judge clause names -> set of POPIA section numbers that are central to that clause.
CLAUSE_TO_SECTIONS: dict[str, list[int]] = {
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


def extract_sections(pdf_path: Path) -> dict[int, dict]:
    """Return {section_number: {title, text}} from the POPIA Act PDF."""
    reader = pypdf.PdfReader(str(pdf_path))
    raw = "\n\n".join(page.extract_text() for page in reader.pages)

    # Each section starts at `^\s*<n>\.\s+<Title>\s*$` on its own line, followed
    # by body text up to the next such marker.
    pattern = re.compile(r"^\s*(\d{1,3})\.\s+([A-Z][^\n]{4,120})$", re.MULTILINE)
    matches = list(pattern.finditer(raw))

    sections: dict[int, dict] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip().replace("ﬁ", "fi").replace("ﬂ", "fl")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        body = re.sub(r"\s+", " ", body)[:2400]
        # Keep the FIRST occurrence per section number (some numbers reappear in cross-refs).
        if num in sections:
            continue
        sections[num] = {"title": title, "text": body}
    return sections


def load_scenarios() -> list[dict]:
    rows = []
    for fname in [
        "popia_seeds.jsonl",
        "popia_paraphrases.jsonl",
        "popia_seeds_v2.jsonl",
        "popia_paraphrases_v2.jsonl",
    ]:
        p = DATA_DIR / fname
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    sections = extract_sections(POPIA_PDF)
    print(f"extracted {len(sections)} POPIA sections")

    scenarios = load_scenarios()
    print(f"loaded {len(scenarios)} labelled scenarios")

    # --- Write the canonical corpus (one row per POPIA section) ---
    with CORPUS_OUT.open("w") as f:
        for num, sec in sorted(sections.items()):
            row = {
                "section_id": f"POPIA-§{num}",
                "title": sec["title"],
                "text": f"POPIA §{num}. {sec['title']}. {sec['text']}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {CORPUS_OUT} ({len(sections)} rows)")

    # --- Write training pairs ---
    pairs: list[dict] = []

    # Pair type 1: (clause name) -> (POPIA section text for any central section).
    for clause, section_nums in CLAUSE_TO_SECTIONS.items():
        for n in section_nums:
            if n not in sections:
                continue
            sec = sections[n]
            pairs.append({
                "anchor": clause,
                "positive": f"POPIA §{n}. {sec['title']}. {sec['text']}",
                "source": "clause-to-section",
            })

    # Pair type 2: (scenario premise) -> (any central section for that scenario's clause).
    #              Only entailment + contradiction scenarios — they reliably target the clause.
    for s in scenarios:
        if s["label"] == "neutral":
            continue
        clause = s["clause"]
        target_sections = CLAUSE_TO_SECTIONS.get(clause, [])
        # Take the FIRST central section per scenario (deterministic; avoids overfitting to one section).
        for n in target_sections[:1]:
            if n not in sections:
                continue
            sec = sections[n]
            pairs.append({
                "anchor": s["premise"],
                "positive": f"POPIA §{n}. {sec['title']}. {sec['text']}",
                "source": f"scenario-{s['scenario']}",
            })

    # Pair type 3: (hypothesis text) -> (central section).
    # The hypothesis is the legal claim; the section is its statutory anchor.
    seen_hyp = set()
    for s in scenarios:
        h = s["hypothesis"]
        if h in seen_hyp:
            continue
        seen_hyp.add(h)
        clause = s["clause"]
        target_sections = CLAUSE_TO_SECTIONS.get(clause, [])
        for n in target_sections[:1]:
            if n not in sections:
                continue
            sec = sections[n]
            pairs.append({
                "anchor": h,
                "positive": f"POPIA §{n}. {sec['title']}. {sec['text']}",
                "source": "hypothesis-to-section",
            })

    with PAIRS_OUT.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {PAIRS_OUT} ({len(pairs)} pairs)")

    # Stats
    from collections import Counter
    src = Counter(p["source"].split("-")[0] if "-" in p["source"] else p["source"] for p in pairs)
    print("pairs by source bucket:", dict(src))


if __name__ == "__main__":
    main()
