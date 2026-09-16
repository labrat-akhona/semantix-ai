#!/usr/bin/env python3
"""Scaffold, validate, and pin data/popia_preset_eval.jsonl.

The preset label set decides whether a POPIA preset may be called validated. Each row pairs one
preset with a premise and the outcome the preset check should reach, labelled against the POPIA
clause (never against the preset's own wording):

    complies      the premise shows the clause being met
    violates      the premise shows the clause being broken
    insufficient  the premise is on topic but does not give enough to decide

Slots are pre-balanced: 10 per preset, split 4 complies / 4 violates / 2 insufficient, spread over
memo, news, and first-person styles. The author writes `scenario` (a short unique tag) and
`premise` for each slot and changes nothing else.

Usage (from the repo root):
    python scripts/validate_popia_preset_eval.py --scaffold   # write the empty slots once
    python scripts/validate_popia_preset_eval.py              # progress and checks
    python scripts/validate_popia_preset_eval.py --pin        # pin SHA-256 when complete and clean

Exit 0: clean (or scaffold written). Exit 1: slots left, errors, or scaffold refused.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

EVAL_PATH = Path("data/popia_preset_eval.jsonl")
HASH_PATH = Path("scripts/_popia_preset_eval_hash.txt")
LEAK_SOURCES = ("data/popia_*.jsonl", "bench/popia-v1/popia_bench.jsonl")

# Mirrors semantix.presets.popia; a unit test keeps the two in sync.
PRESET_CLAUSES = {
    "POPIA_CONSENT": "POPIA consent",
    "POPIA_MINIMALITY": "POPIA minimality / purpose limitation",
    "POPIA_SECURITY": "POPIA security safeguards",
    "POPIA_BREACH": "POPIA breach notification",
    "POPIA_CROSS_BORDER": "POPIA cross-border transfers",
    "POPIA_PROCESSING": "POPIA general processing",
    "POPIA_DATA_SUBJECT_RIGHTS": "POPIA data subject rights",
}
SPLIT = {"complies": 4, "violates": 4, "insufficient": 2}
PER_PRESET = sum(SPLIT.values())
STYLES = ("memo", "news", "first-person")
MIN_PER_STYLE = 2
SHORT_PREMISE_CHARS = 60
KEYS = {"preset", "clause", "expected", "style", "scenario", "premise"}
TODO = "TODO"


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def scaffold_rows() -> list[dict]:
    """The empty slots: labels and styles pre-assigned, scenario and premise left TODO."""
    labels = [label for label, count in SPLIT.items() for _ in range(count)]
    return [
        {
            "preset": preset,
            "clause": clause,
            "expected": expected,
            "style": STYLES[i % len(STYLES)],
            "scenario": TODO,
            "premise": TODO,
        }
        for preset, clause in PRESET_CLAUSES.items()
        for i, expected in enumerate(labels)
    ]


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_leak_premises() -> set[str]:
    """Normalised premises from every existing POPIA data file and POPIA-Bench."""
    premises: set[str] = set()
    for pattern in LEAK_SOURCES:
        for name in glob.glob(pattern):
            if Path(name).resolve() == Path(EVAL_PATH).resolve():
                continue
            for row in load_rows(Path(name)):
                if isinstance(row, dict) and row.get("premise"):
                    premises.add(norm(row["premise"]))
    return premises


@dataclass
class Report:
    todo: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.todo and not self.errors


def validate(rows: list[dict], leak_premises: set[str]) -> Report:
    report = Report()
    report.todo = [
        i
        for i, r in enumerate(rows)
        if any(TODO in str(r.get(key, "")) for key in ("scenario", "premise"))
    ]

    expected_rows = len(PRESET_CLAUSES) * PER_PRESET
    if len(rows) != expected_rows:
        report.errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    for i, r in enumerate(rows):
        if set(r) != KEYS:
            report.errors.append(f"row {i}: keys must be {sorted(KEYS)}, found {sorted(r)}")
            continue
        if r["preset"] not in PRESET_CLAUSES:
            report.errors.append(f"row {i}: unknown preset {r['preset']!r}")
        elif r["clause"] != PRESET_CLAUSES[r["preset"]]:
            report.errors.append(f"row {i}: clause {r['clause']!r} does not match {r['preset']}")
        if r["expected"] not in SPLIT:
            report.errors.append(f"row {i}: expected must be one of {sorted(SPLIT)}")
        if r["style"] not in STYLES:
            report.errors.append(f"row {i}: style must be one of {list(STYLES)}")

    well_formed = [r for r in rows if set(r) == KEYS]
    for preset in PRESET_CLAUSES:
        mine = [r for r in well_formed if r["preset"] == preset]
        split = dict(collections.Counter(r["expected"] for r in mine))
        if split != SPLIT:
            report.errors.append(f"{preset}: label split {split}, want {SPLIT}")
        styles = collections.Counter(r["style"] for r in mine)
        thin = [s for s in STYLES if styles.get(s, 0) < MIN_PER_STYLE]
        if thin:
            report.errors.append(f"{preset}: fewer than {MIN_PER_STYLE} premises in {thin}")

    todo = set(report.todo)
    seen_premise: dict[str, int] = {}
    seen_scenario: dict[str, int] = {}
    for i, r in enumerate(rows):
        if i in todo or set(r) != KEYS:
            continue
        premise, scenario = norm(r["premise"]), norm(r["scenario"])
        if premise in seen_premise:
            report.errors.append(f"row {i}: premise duplicates row {seen_premise[premise]}")
        seen_premise.setdefault(premise, i)
        if scenario in seen_scenario:
            report.errors.append(f"row {i}: scenario tag duplicates row {seen_scenario[scenario]}")
        seen_scenario.setdefault(scenario, i)
        if premise in leak_premises:
            report.errors.append(
                f"row {i}: premise already exists in POPIA data or POPIA-Bench (leak)"
            )
        if len(r["premise"].strip()) < SHORT_PREMISE_CHARS:
            report.warnings.append(
                f"row {i}: premise under {SHORT_PREMISE_CHARS} characters; the one-line preset "
                f"examples in the 2026-09-15 review scored near zero, so give concrete detail"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scaffold, validate, and pin the preset label set.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--scaffold", action="store_true", help="write empty slots (never overwrites)"
    )
    mode.add_argument("--pin", action="store_true", help="pin SHA-256 if complete and clean")
    args = ap.parse_args(argv)
    eval_path, hash_path = Path(EVAL_PATH), Path(HASH_PATH)

    if args.scaffold:
        if eval_path.exists():
            print(f"{eval_path} already exists; refusing to overwrite authored work.")
            return 1
        rows = scaffold_rows()
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} slots to {eval_path}")
        return 0

    if not eval_path.exists():
        print(f"missing {eval_path}; run with --scaffold first")
        return 1

    rows = load_rows(eval_path)
    report = validate(rows, load_leak_premises())
    left_by_preset = collections.Counter(rows[i].get("preset") for i in report.todo)
    print(f"authored {len(rows) - len(report.todo)}/{len(rows)}")
    for preset in PRESET_CLAUSES:
        left = left_by_preset.get(preset, 0)
        done = PER_PRESET - left
        print(f"  [{'#' * done}{'.' * left}] {done}/{PER_PRESET}  {preset}")
    for warning in report.warnings:
        print(f"  WARN  {warning}")
    for error in report.errors:
        print(f"  ERROR {error}")

    if not report.clean:
        if args.pin:
            print("not pinned: finish every slot and clear every error first")
        return 1

    digest = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    print(f"CLEAN{' (with warnings)' if report.warnings else ''}  sha256: {digest}")
    if args.pin:
        hash_path.write_text(digest + "\n", encoding="utf-8")
        print(f"pinned -> {hash_path}. Commit the hash on its own so reviewers can audit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
