"""Threshold calibration from training data."""

from __future__ import annotations

import json
from pathlib import Path


def calibrate_thresholds(
    training_path: str | Path,
) -> dict[str, float]:
    """Compute optimal per-intent thresholds from TrainingCollector data.

    For each intent, the threshold is set to the midpoint between the
    highest rejected score and the lowest accepted score. This maximizes
    the separation between pass/fail examples.

    Parameters
    ----------
    training_path:
        Path to a TrainingCollector JSONL file.

    Returns
    -------
    dict[str, float]
        Mapping of intent name → calibrated threshold.
    """
    path = Path(training_path)
    if not path.exists():
        return {}

    intent_scores: dict[str, dict[str, list[float]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record["intent"]
            if name not in intent_scores:
                intent_scores[name] = {"rejected": [], "accepted": []}
            if record.get("rejected_score") is not None:
                intent_scores[name]["rejected"].append(record["rejected_score"])
            if record.get("accepted_score") is not None:
                intent_scores[name]["accepted"].append(record["accepted_score"])

    thresholds: dict[str, float] = {}
    for name, scores in intent_scores.items():
        if not scores["rejected"] or not scores["accepted"]:
            continue
        max_rejected = max(scores["rejected"])
        min_accepted = min(scores["accepted"])
        if min_accepted > max_rejected:
            # Clean separation — midpoint
            thresholds[name] = (max_rejected + min_accepted) / 2
        else:
            # Overlap — use average as rough boundary
            all_scores = scores["rejected"] + scores["accepted"]
            thresholds[name] = sum(all_scores) / len(all_scores)

    return thresholds


def apply_calibration(
    thresholds: dict[str, float],
    intent_classes: dict[str, type],
) -> int:
    """Apply calibrated thresholds to Intent classes.

    Parameters
    ----------
    thresholds:
        Output of calibrate_thresholds().
    intent_classes:
        Mapping of intent name → Intent subclass.

    Returns
    -------
    int
        Number of intents that were calibrated.
    """
    applied = 0
    for name, threshold in thresholds.items():
        if name in intent_classes:
            intent_classes[name].threshold = threshold
            applied += 1
    return applied
