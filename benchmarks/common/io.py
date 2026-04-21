"""CSV and summary-markdown writers for benchmark results."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Row:
    example_id: str
    experiment: str  # "agreement" or "optimization"
    judge: str
    intent: str
    text: str
    score: float
    latency_ms: float
    cost_usd: float
    paid_equivalent_usd: float
    raw: str | None
    error: str | None


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(Row)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: getattr(row, fn) for fn in fieldnames})


def write_summary_md(rows: list[Row], path: Path, *, task_name: str) -> None:
    """Write headline comparison table grouped by judge."""
    by_judge: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_judge[r.judge].append(r)

    def _mean(xs: list[float]) -> float:
        xs = [x for x in xs if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else float("nan")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {task_name}",
        "",
        "## Headline comparison",
        "",
        "| Judge | Avg latency (ms) | Free-tier cost/1k | Paid-tier cost/1k | Error rate |",
        "|---|---|---|---|---|",
    ]
    for judge, judge_rows in sorted(by_judge.items()):
        latencies = [r.latency_ms for r in judge_rows if r.error is None]
        errors = sum(1 for r in judge_rows if r.error is not None)
        total = len(judge_rows)
        paid_cost_per_1k = _mean([r.paid_equivalent_usd for r in judge_rows]) * 1000
        free_cost_per_1k = _mean([r.cost_usd for r in judge_rows]) * 1000
        lines.append(
            f"| {judge} | {_mean(latencies):.0f} | ${free_cost_per_1k:.4f} | "
            f"${paid_cost_per_1k:.4f} | {errors}/{total} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
