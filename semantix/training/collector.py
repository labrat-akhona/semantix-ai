"""TrainingCollector — append-only JSONL storage for correction pairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TrainingCollector:
    """Collects (rejected, accepted) training pairs from self-healing retries.

    Each call to ``record()`` appends one JSONL line to disk.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The JSONL file path."""
        return self._path

    def record(
        self,
        *,
        intent: str,
        intent_description: str,
        rejected_output: str,
        rejected_score: float | None,
        rejected_reason: str | None,
        accepted_output: str,
        accepted_score: float | None,
        feedback: str,
        attempts: int,
    ) -> None:
        """Append a training pair to the JSONL file."""
        entry = {
            "intent": intent,
            "intent_description": intent_description,
            "rejected_output": rejected_output,
            "rejected_score": rejected_score,
            "rejected_reason": rejected_reason,
            "accepted_output": accepted_output,
            "accepted_score": accepted_score,
            "feedback": feedback,
            "attempts": attempts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def stats(self) -> dict:
        """Read the JSONL file and return aggregate counts."""
        if not self._path.exists():
            return {"total_pairs": 0, "intents": {}}
        intents: dict[str, int] = {}
        total = 0
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total += 1
                name = data["intent"]
                intents[name] = intents.get(name, 0) + 1
        return {"total_pairs": total, "intents": intents}
