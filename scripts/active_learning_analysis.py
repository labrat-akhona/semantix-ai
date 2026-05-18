"""Active-learning evaluation loop for Semantix POPIA models.

Runs the existing POPIA eval, captures **per-example** results, then performs:

  1. Error analysis   – failures by clause, error type, complexity, confidence
  2. Gap analysis     – training-data coverage vs. the full POPIA Act
  3. Priority scoring – ranks every gap by impact × frequency × weakness
  4. Data-gen brief   – structured JSON prescription for the next training batch

Usage
-----
    # Full run (needs the ONNX models — will download on first use)
    python scripts/active_learning_analysis.py

    # Use HF-hosted eval.jsonl instead of local
    python scripts/active_learning_analysis.py --use-hf

    # Custom output path
    python scripts/active_learning_analysis.py -o reports/next_batch.json

    # Dry-run: skip model inference, use synthetic scores (for CI / dev)
    python scripts/active_learning_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 0.  POPIA reference data — canonical structure of the Act
# ---------------------------------------------------------------------------

# Complete POPIA chapter/section map used for gap analysis.
# Each entry maps a *clause label* (the concept the NLI model predicts on)
# to the POPIA sections it covers, a short description, and a rough
# real-world frequency weight (how often this clause surfaces in compliance
# work — expert-estimated, 1-10 scale).

POPIA_SECTIONS: dict[str, dict[str, Any]] = {
    "POPIA consent": {
        "sections": ["§9", "§11", "§11(1)(a)", "§11(3)", "§69", "§69(2)"],
        "chapter": 3,
        "description": "Consent requirements for lawful processing and direct marketing",
        "real_world_frequency": 10,
    },
    "POPIA minimality / purpose limitation": {
        "sections": ["§9", "§10", "§13", "§14", "§15"],
        "chapter": 3,
        "description": "Conditions 2-5: purpose specification, further processing, minimality, retention",
        "real_world_frequency": 9,
    },
    "POPIA general processing": {
        "sections": ["§8", "§9", "§10", "§11", "§12"],
        "chapter": 3,
        "description": "Condition 1: accountability and general lawfulness of processing",
        "real_world_frequency": 8,
    },
    "POPIA security safeguards": {
        "sections": ["§19", "§20", "§21", "§22"],
        "chapter": 3,
        "description": "Condition 7: security measures, breach notification to Regulator and subjects",
        "real_world_frequency": 9,
    },
    "POPIA breach notification": {
        "sections": ["§22"],
        "chapter": 3,
        "description": "Notification of security compromises — timing, content, recipients",
        "real_world_frequency": 8,
    },
    "POPIA data subject rights": {
        "sections": ["§5", "§23", "§24", "§25"],
        "chapter": 3,
        "description": "Condition 8: data-subject participation — access, correction, deletion",
        "real_world_frequency": 9,
    },
    "POPIA cross-border transfers": {
        "sections": ["§72"],
        "chapter": 9,
        "description": "Transborder information flows — adequacy, consent, binding rules",
        "real_world_frequency": 7,
    },
    # V2 clauses (may or may not be in the current training / eval data)
    "POPIA children's information": {
        "sections": ["§34", "§35"],
        "chapter": 3,
        "description": "Processing of personal information of children under 18",
        "real_world_frequency": 5,
    },
    "POPIA special personal information": {
        "sections": ["§26", "§27", "§28", "§29", "§30", "§31", "§32", "§33"],
        "chapter": 3,
        "description": "Processing of special PI — race, health, biometrics, criminal, etc.",
        "real_world_frequency": 6,
    },
    "POPIA automated decision-making": {
        "sections": ["§71"],
        "chapter": 8,
        "description": "Rights regarding automated processing and profiling",
        "real_world_frequency": 6,
    },
    # Sections with NO dedicated clause label yet (pure gap)
    "POPIA exemptions": {
        "sections": ["§36", "§37", "§38"],
        "chapter": 4,
        "description": "Exemptions from processing conditions — journalism, art, research",
        "real_world_frequency": 4,
    },
    "POPIA information officers": {
        "sections": ["§55", "§56"],
        "chapter": 5,
        "description": "Designation, registration, and duties of information officers",
        "real_world_frequency": 6,
    },
    "POPIA prior authorisation": {
        "sections": ["§57", "§58", "§59"],
        "chapter": 6,
        "description": "Prior authorisation for high-risk processing (e.g. unique identifiers, criminal)",
        "real_world_frequency": 4,
    },
    "POPIA codes of conduct": {
        "sections": ["§60", "§61", "§62", "§63", "§64", "§65", "§66", "§67", "§68"],
        "chapter": 7,
        "description": "Industry codes of conduct — application, approval, enforcement",
        "real_world_frequency": 3,
    },
    "POPIA direct marketing": {
        "sections": ["§69", "§69(1)", "§69(2)", "§69(3)", "§70"],
        "chapter": 8,
        "description": "Unsolicited electronic communications — opt-in/opt-out rules",
        "real_world_frequency": 8,
    },
    "POPIA directories": {
        "sections": ["§70"],
        "chapter": 8,
        "description": "Rights regarding directories and automated decision-making",
        "real_world_frequency": 3,
    },
    "POPIA enforcement": {
        "sections": ["§73", "§74", "§75", "§76", "§77", "§78", "§99"],
        "chapter": 10,
        "description": "Complaints, investigations, enforcement notices, civil remedies",
        "real_world_frequency": 5,
    },
    "POPIA offences and penalties": {
        "sections": ["§100", "§101", "§102", "§103", "§104", "§105", "§106", "§107"],
        "chapter": 11,
        "description": "Criminal offences, penalties, administrative fines",
        "real_world_frequency": 5,
    },
    "POPIA openness (notification)": {
        "sections": ["§16", "§17", "§18"],
        "chapter": 3,
        "description": "Condition 6: notification to the Regulator and to data subjects",
        "real_world_frequency": 6,
    },
    "POPIA information quality": {
        "sections": ["§16"],
        "chapter": 3,
        "description": "Condition 4 (quality): information must be complete, accurate, up to date",
        "real_world_frequency": 5,
    },
}

# Scenario complexity heuristics — keywords that indicate multi-hop reasoning.
_MULTI_HOP_KEYWORDS = [
    "cross-border", "multi-", "exception", "exemption", "conflict",
    "override", "despite", "notwithstanding", "however", "unless",
    "both", "combined", "interaction", "together with", "in addition to",
    "simultaneously", "competing", "multiple", "two or more",
]


# ---------------------------------------------------------------------------
# 1.  Per-example evaluation
# ---------------------------------------------------------------------------

@dataclass
class ExampleResult:
    """Result for a single eval example."""
    idx: int
    clause: str
    label: str                     # ground truth: entailment | contradiction | neutral
    scenario: str
    premise: str
    hypothesis: str
    popia_passed: bool             # model predicted entailment?
    popia_score: float             # entailment probability
    stock_passed: bool
    stock_score: float
    correct: bool                  # did popia judge agree with ground truth?
    error_type: str                # ok | false_positive | false_negative | wrong_neutral
    complexity: str                # simple | multi_hop
    confidence_bucket: str         # high_correct | high_wrong | low_correct | low_wrong


@dataclass
class AnalysisReport:
    """Complete analysis output."""
    summary: dict[str, Any] = field(default_factory=dict)
    per_example: list[dict[str, Any]] = field(default_factory=list)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    gap_analysis: dict[str, Any] = field(default_factory=dict)
    priority_scores: list[dict[str, Any]] = field(default_factory=list)
    data_generation_brief: list[dict[str, Any]] = field(default_factory=list)


def _classify_error(label: str, popia_passed: bool) -> str:
    """Classify the error type for a single example."""
    truth_is_entailment = label == "entailment"
    if popia_passed == truth_is_entailment:
        # For neutral/contradiction ground-truth: model should NOT pass.
        # For entailment: model SHOULD pass.
        if label in ("contradiction", "neutral") and not popia_passed:
            return "ok"
        if label == "entailment" and popia_passed:
            return "ok"
        return "ok"

    # Mismatches
    if popia_passed and label == "contradiction":
        return "false_positive"           # model said entailment, truth is contradiction
    if popia_passed and label == "neutral":
        return "false_positive_neutral"   # model said entailment, truth is neutral
    if not popia_passed and label == "entailment":
        return "false_negative"           # model said not-entailment, truth is entailment
    return "unknown_error"


def _classify_complexity(scenario: str, premise: str) -> str:
    """Heuristic: is this a simple or multi-hop scenario?"""
    text = (scenario + " " + premise).lower()
    hits = sum(1 for kw in _MULTI_HOP_KEYWORDS if kw in text)
    return "multi_hop" if hits >= 1 else "simple"


def _confidence_bucket(score: float, correct: bool) -> str:
    """Bin by confidence × correctness."""
    high = score >= 0.75 or score <= 0.25  # model is decisive
    if correct:
        return "high_correct" if high else "low_correct"
    else:
        return "high_wrong" if high else "low_wrong"


def run_eval(
    eval_path: Path,
    dry_run: bool = False,
) -> list[ExampleResult]:
    """Run both judges on every eval example and return per-example results."""
    rows = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]

    if dry_run:
        # Synthetic scores for CI/dev — no model download needed.
        import random
        random.seed(42)
        results: list[ExampleResult] = []
        for i, r in enumerate(rows):
            score = random.random()
            passed = score >= 0.5
            truth_entailment = r["label"] == "entailment"
            correct = passed == truth_entailment if r["label"] != "neutral" else not passed
            results.append(ExampleResult(
                idx=i, clause=r["clause"], label=r["label"],
                scenario=r.get("scenario", ""), premise=r["premise"],
                hypothesis=r["hypothesis"],
                popia_passed=passed, popia_score=score,
                stock_passed=random.random() >= 0.5,
                stock_score=random.random(),
                correct=correct,
                error_type=_classify_error(r["label"], passed),
                complexity=_classify_complexity(r.get("scenario", ""), r["premise"]),
                confidence_bucket=_confidence_bucket(score, correct),
            ))
        return results

    # Real inference
    from semantix.judges.popia import POPIAJudge
    from semantix.judges.quantized_nli import QuantizedNLIJudge

    popia_judge = POPIAJudge()
    stock_judge = QuantizedNLIJudge()
    results = []

    for i, r in enumerate(rows):
        pv = popia_judge.evaluate(r["premise"], r["hypothesis"])
        sv = stock_judge.evaluate(r["premise"], r["hypothesis"])

        truth_entailment = r["label"] == "entailment"
        # For non-entailment labels the model should NOT pass
        if r["label"] == "entailment":
            correct = pv.passed
        else:
            correct = not pv.passed

        results.append(ExampleResult(
            idx=i,
            clause=r["clause"],
            label=r["label"],
            scenario=r.get("scenario", ""),
            premise=r["premise"],
            hypothesis=r["hypothesis"],
            popia_passed=pv.passed,
            popia_score=pv.score or 0.0,
            stock_passed=sv.passed,
            stock_score=sv.score or 0.0,
            correct=correct,
            error_type=_classify_error(r["label"], pv.passed),
            complexity=_classify_complexity(r.get("scenario", ""), r["premise"]),
            confidence_bucket=_confidence_bucket(pv.score or 0.0, correct),
        ))

    return results


# ---------------------------------------------------------------------------
# 2.  Error analysis
# ---------------------------------------------------------------------------

def error_analysis(results: list[ExampleResult]) -> dict[str, Any]:
    """Categorise failures by clause, error type, complexity, confidence."""
    total = len(results)
    failures = [r for r in results if not r.correct]

    # --- By clause ---
    clause_errors: dict[str, list[ExampleResult]] = defaultdict(list)
    clause_totals: Counter[str] = Counter()
    for r in results:
        clause_totals[r.clause] += 1
        if not r.correct:
            clause_errors[r.clause].append(r)

    by_clause = {}
    for clause in sorted(clause_totals):
        errs = clause_errors.get(clause, [])
        by_clause[clause] = {
            "total": clause_totals[clause],
            "errors": len(errs),
            "error_rate": round(len(errs) / clause_totals[clause], 4) if clause_totals[clause] else 0,
            "error_types": dict(Counter(e.error_type for e in errs)),
        }

    # --- By error type ---
    by_error_type = dict(Counter(r.error_type for r in results))

    # --- By complexity ---
    complexity_acc: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        complexity_acc[r.complexity]["total"] += 1
        if r.correct:
            complexity_acc[r.complexity]["correct"] += 1
    by_complexity = {
        k: {
            **v,
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
        }
        for k, v in complexity_acc.items()
    }

    # --- Confidence calibration ---
    bucket_counts = dict(Counter(r.confidence_bucket for r in results))
    # Also compute mean score for wrong answers per clause
    confidently_wrong = [r for r in failures if r.confidence_bucket == "high_wrong"]

    return {
        "total_examples": total,
        "total_errors": len(failures),
        "overall_accuracy": round((total - len(failures)) / total, 4) if total else 0,
        "by_clause": by_clause,
        "by_error_type": by_error_type,
        "by_complexity": by_complexity,
        "confidence_buckets": bucket_counts,
        "confidently_wrong_count": len(confidently_wrong),
        "confidently_wrong_examples": [
            {
                "idx": r.idx,
                "clause": r.clause,
                "label": r.label,
                "score": round(r.popia_score, 4),
                "scenario": r.scenario,
                "premise_preview": r.premise[:120],
            }
            for r in confidently_wrong[:20]  # cap for readability
        ],
    }


# ---------------------------------------------------------------------------
# 3.  Gap analysis
# ---------------------------------------------------------------------------

def _load_training_data(data_dir: Path) -> list[dict]:
    """Load all POPIA training JSONL files (seeds + paraphrases)."""
    training_files = [
        "popia_seeds.jsonl",
        "popia_seeds_v2.jsonl",
        "popia_paraphrases.jsonl",
        "popia_paraphrases_v2.jsonl",
    ]
    rows: list[dict] = []
    for fname in training_files:
        p = data_dir / fname
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def gap_analysis(
    results: list[ExampleResult],
    data_dir: Path,
) -> dict[str, Any]:
    """Compare training-data coverage against full POPIA Act."""
    training = _load_training_data(data_dir)

    # Training distribution
    train_clause_counts: Counter[str] = Counter()
    train_label_counts: Counter[tuple[str, str]] = Counter()
    train_scenarios: dict[str, set[str]] = defaultdict(set)
    for t in training:
        c = t.get("clause", "")
        train_clause_counts[c] += 1
        train_label_counts[(c, t.get("label", ""))] += 1
        train_scenarios[c].add(t.get("scenario", ""))

    # Eval distribution
    eval_clause_counts: Counter[str] = Counter()
    eval_scenarios: dict[str, set[str]] = defaultdict(set)
    for r in results:
        eval_clause_counts[r.clause] += 1
        eval_scenarios[r.clause].add(r.scenario)

    # --- Sections with zero/few training examples ---
    coverage: dict[str, dict[str, Any]] = {}
    for clause, meta in POPIA_SECTIONS.items():
        train_n = train_clause_counts.get(clause, 0)
        eval_n = eval_clause_counts.get(clause, 0)
        coverage[clause] = {
            "popia_sections": meta["sections"],
            "chapter": meta["chapter"],
            "description": meta["description"],
            "training_examples": train_n,
            "eval_examples": eval_n,
            "training_scenarios": sorted(train_scenarios.get(clause, set())),
            "coverage_status": (
                "none" if train_n == 0
                else "sparse" if train_n < 10
                else "moderate" if train_n < 25
                else "good"
            ),
        }

    # --- Missing clause × label combinations ---
    all_labels = ["entailment", "contradiction", "neutral"]
    trained_clauses = set(train_clause_counts.keys())
    missing_combos: list[dict[str, str]] = []
    for clause in POPIA_SECTIONS:
        for lbl in all_labels:
            if train_label_counts.get((clause, lbl), 0) == 0:
                missing_combos.append({"clause": clause, "label": lbl})

    # --- Missing scenario types ---
    # Derive desired scenario archetypes
    desired_archetypes = [
        "first-person",       # data-subject perspective
        "news",               # third-party press report
        "memo",               # internal corporate memo
        "audit",              # audit / investigation finding
        "complaint",          # formal complaint to Regulator
        "contract",           # contractual clause / T&C excerpt
        "multi-party",        # scenarios involving >2 entities
        "cross-border",       # international transfer dimension
        "technical",          # IT / cybersecurity angle
        "healthcare",         # health-sector specific
        "financial",          # FICA / banking sector
        "education",          # schools / universities
        "employment",         # employer-employee relationship
    ]
    all_train_scenarios = set()
    for s in train_scenarios.values():
        all_train_scenarios |= s

    archetype_coverage: dict[str, int] = {}
    for arch in desired_archetypes:
        archetype_coverage[arch] = sum(
            1 for s in all_train_scenarios if arch in s.lower()
        )

    missing_archetypes = [a for a, c in archetype_coverage.items() if c == 0]

    return {
        "training_total": len(training),
        "clause_coverage": coverage,
        "untested_clause_label_combos": missing_combos,
        "scenario_archetype_coverage": archetype_coverage,
        "missing_scenario_archetypes": missing_archetypes,
        "clauses_with_zero_training": [
            c for c, v in coverage.items() if v["training_examples"] == 0
        ],
        "clauses_with_sparse_training": [
            c for c, v in coverage.items() if v["coverage_status"] == "sparse"
        ],
    }


# ---------------------------------------------------------------------------
# 4.  Priority scoring
# ---------------------------------------------------------------------------

def priority_scoring(
    err: dict[str, Any],
    gaps: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank data gaps by impact, frequency, and current weakness.

    Score = impact × frequency × weakness   (each normalised to 0-1)

    - impact:    estimated F1 lift from fixing this clause ∝ error rate × volume share
    - frequency: how often this clause arises in real compliance work (from POPIA_SECTIONS)
    - weakness:  1 - current_accuracy for this clause (worst-performing → highest score)
    """
    clause_coverage = gaps["clause_coverage"]
    by_clause = err["by_clause"]
    total_examples = err["total_examples"]

    scored: list[dict[str, Any]] = []

    for clause, meta in POPIA_SECTIONS.items():
        stats = by_clause.get(clause, {})
        cov = clause_coverage.get(clause, {})

        clause_total = stats.get("total", 0)
        clause_errors = stats.get("errors", 0)
        error_rate = stats.get("error_rate", 0.0)
        train_n = cov.get("training_examples", 0)

        # Impact: proportional to how much fixing this clause would move overall F1
        # Approximate as (errors_in_clause / total_examples) — the share of total
        # errors attributable to this clause, weighted by how trainable the gap is.
        if total_examples > 0:
            impact = (clause_errors / total_examples) + (0.1 if train_n == 0 else 0)
        else:
            impact = 0.1 if train_n == 0 else 0.0

        # Frequency: normalised real-world frequency (1-10 → 0.1-1.0)
        freq = meta["real_world_frequency"] / 10.0

        # Weakness: 1 - accuracy for this clause
        if clause_total > 0:
            weakness = error_rate
        else:
            # No eval data → assume moderately weak
            weakness = 0.5 if train_n == 0 else 0.3

        # Composite priority
        priority = round(impact * freq * weakness * 1000, 2)  # scale for readability

        # Estimate how many examples are needed
        if train_n == 0:
            suggested_count = 30   # bootstrap a brand-new clause
        elif train_n < 10:
            suggested_count = max(20, 30 - train_n)
        elif error_rate > 0.3:
            suggested_count = 15   # targeted patch
        elif error_rate > 0.15:
            suggested_count = 10
        else:
            suggested_count = 5    # maintenance

        scored.append({
            "clause": clause,
            "priority_score": priority,
            "impact": round(impact, 4),
            "frequency": round(freq, 2),
            "weakness": round(weakness, 4),
            "current_training_examples": train_n,
            "current_eval_accuracy": round(1 - error_rate, 4) if clause_total > 0 else None,
            "suggested_additional_examples": suggested_count,
            "popia_sections": meta["sections"],
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 5.  Data-generation brief
# ---------------------------------------------------------------------------

def data_generation_brief(
    priorities: list[dict[str, Any]],
    gaps: dict[str, Any],
    err: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce an actionable JSON brief: what to generate next."""
    missing_archetypes = gaps.get("missing_scenario_archetypes", [])
    untested_combos = gaps.get("untested_clause_label_combos", [])
    by_clause_errors = err.get("by_clause", {})

    # Build a lookup for untested label combos per clause
    untested_labels: dict[str, list[str]] = defaultdict(list)
    for combo in untested_combos:
        untested_labels[combo["clause"]].append(combo["label"])

    brief: list[dict[str, Any]] = []

    for p in priorities:
        clause = p["clause"]
        if p["priority_score"] <= 0 and p["current_training_examples"] > 10:
            continue  # skip well-covered, high-performing clauses

        # Determine which error types are most prevalent for this clause
        clause_err = by_clause_errors.get(clause, {})
        dominant_errors = clause_err.get("error_types", {})

        # Suggest scenario types
        suggested_scenarios: list[str] = []
        # Always include missing archetypes for under-covered clauses
        if p["current_training_examples"] < 15:
            suggested_scenarios.extend(missing_archetypes[:3])
        # If false positives dominate, we need more contradiction examples
        if dominant_errors.get("false_positive", 0) + dominant_errors.get("false_positive_neutral", 0) > dominant_errors.get("false_negative", 0):
            suggested_scenarios.append("hard-negative (contradiction)")
            suggested_scenarios.append("subtle-neutral")
        elif dominant_errors.get("false_negative", 0) > 0:
            suggested_scenarios.append("clear-entailment")
            suggested_scenarios.append("entailment-with-noise")

        # Determine difficulty
        weakness = p["weakness"]
        if weakness > 0.4:
            difficulty = "mixed (50% basic, 30% intermediate, 20% hard)"
        elif weakness > 0.2:
            difficulty = "intermediate-to-hard"
        elif p["current_training_examples"] == 0:
            difficulty = "progressive (start basic, ramp up)"
        else:
            difficulty = "hard (edge cases and exceptions)"

        # Which labels to focus on
        focus_labels = untested_labels.get(clause, [])
        if not focus_labels:
            # Balance based on dominant error types
            if dominant_errors.get("false_positive", 0) > 0:
                focus_labels = ["contradiction", "neutral"]
            elif dominant_errors.get("false_negative", 0) > 0:
                focus_labels = ["entailment"]
            else:
                focus_labels = ["entailment", "contradiction", "neutral"]

        brief.append({
            "target_clauses": [clause],
            "popia_sections": p["popia_sections"],
            "suggested_scenario_types": list(dict.fromkeys(suggested_scenarios))[:6],
            "focus_labels": focus_labels,
            "desired_difficulty": difficulty,
            "expected_count": p["suggested_additional_examples"],
            "priority_score": p["priority_score"],
            "rationale": _brief_rationale(p, clause_err, gaps["clause_coverage"].get(clause, {})),
        })

    # Sort by priority
    brief.sort(key=lambda x: x["priority_score"], reverse=True)
    return brief


def _brief_rationale(
    priority: dict[str, Any],
    clause_err: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    """One-line human-readable rationale for this brief entry."""
    parts: list[str] = []
    train_n = priority["current_training_examples"]
    acc = priority["current_eval_accuracy"]

    if train_n == 0:
        parts.append("ZERO training examples — completely untested clause")
    elif train_n < 10:
        parts.append(f"only {train_n} training examples (sparse)")

    if acc is not None and acc < 0.7:
        parts.append(f"low accuracy ({acc:.0%})")
    elif acc is not None and acc < 0.85:
        parts.append(f"moderate accuracy ({acc:.0%})")

    freq = priority["frequency"]
    if freq >= 0.8:
        parts.append("high real-world frequency")

    errs = clause_err.get("error_types", {})
    if errs.get("false_positive", 0) > errs.get("false_negative", 0):
        parts.append("dominated by false positives")
    elif errs.get("false_negative", 0) > 0:
        parts.append("false negatives present")

    return "; ".join(parts) if parts else "maintenance coverage"


# ---------------------------------------------------------------------------
# 6.  Main orchestrator
# ---------------------------------------------------------------------------

def build_report(
    eval_path: Path,
    data_dir: Path,
    dry_run: bool = False,
) -> AnalysisReport:
    """Run the full active-learning analysis pipeline."""
    print("▸ Running per-example evaluation …")
    results = run_eval(eval_path, dry_run=dry_run)
    n_correct = sum(1 for r in results if r.correct)
    print(f"  {len(results)} examples, {n_correct} correct ({n_correct/len(results):.1%})")

    print("▸ Error analysis …")
    err = error_analysis(results)

    print("▸ Gap analysis …")
    gaps = gap_analysis(results, data_dir)

    print("▸ Priority scoring …")
    priorities = priority_scoring(err, gaps)

    print("▸ Generating data-collection brief …")
    brief = data_generation_brief(priorities, gaps, err)

    # Summary
    summary = {
        "eval_file": str(eval_path),
        "total_eval_examples": len(results),
        "overall_accuracy": err["overall_accuracy"],
        "total_training_examples": gaps["training_total"],
        "clauses_with_zero_training": gaps["clauses_with_zero_training"],
        "clauses_with_sparse_training": gaps["clauses_with_sparse_training"],
        "top_3_priorities": [
            {"clause": p["clause"], "score": p["priority_score"]}
            for p in priorities[:3]
        ],
        "total_examples_to_generate": sum(b["expected_count"] for b in brief),
    }

    return AnalysisReport(
        summary=summary,
        per_example=[asdict(r) for r in results],
        error_analysis=err,
        gap_analysis=gaps,
        priority_scores=priorities,
        data_generation_brief=brief,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Active-learning analysis for POPIA NLI model",
    )
    ap.add_argument(
        "--use-hf", action="store_true",
        help="Download eval.jsonl from HuggingFace instead of using local file.",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        default=Path("reports/active_learning_brief.json"),
        help="Path for the JSON output (default: reports/active_learning_brief.json).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Skip model inference — use synthetic scores for testing the pipeline.",
    )
    args = ap.parse_args()

    # Resolve eval path
    local_eval = Path("data/popia_eval.jsonl")
    if args.use_hf:
        from huggingface_hub import hf_hub_download
        eval_path = Path(hf_hub_download(
            repo_id="labrat-aiko/nli-popia-v1", filename="eval.jsonl",
        ))
    else:
        if not local_eval.exists():
            print(f"✗ Missing {local_eval} — run from repo root or use --use-hf", file=sys.stderr)
            return 2
        eval_path = local_eval

    data_dir = Path("data")

    report = build_report(eval_path, data_dir, dry_run=args.dry_run)

    # Write full report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n✓ Full report written to {args.output}")

    # Also write the brief separately for easy consumption
    brief_path = args.output.with_name("data_generation_brief.json")
    with open(brief_path, "w") as f:
        json.dump(report.data_generation_brief, f, indent=2)
    print(f"✓ Data-generation brief written to {brief_path}")

    # Print top priorities to stdout
    print("\n" + "=" * 70)
    print("TOP PRIORITIES — where to invest next training data")
    print("=" * 70)
    for i, item in enumerate(report.data_generation_brief[:10], 1):
        print(
            f"\n  {i}. {item['target_clauses'][0]}"
            f"\n     Priority: {item['priority_score']:.1f}  |  "
            f"Generate: {item['expected_count']} examples  |  "
            f"Difficulty: {item['desired_difficulty']}"
            f"\n     Rationale: {item['rationale']}"
        )

    print(f"\n  TOTAL examples to generate: {report.summary['total_examples_to_generate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
