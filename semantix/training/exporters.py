"""Export training data in various fine-tuning formats."""

from __future__ import annotations

import json
from pathlib import Path


def _read_records(source: Path, intent_filter: str | None = None) -> list[dict]:
    """Read JSONL records, optionally filtering by intent name."""
    records = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if intent_filter and data["intent"] != intent_filter:
                continue
            records.append(data)
    return records


def export_generic(
    source: str | Path,
    destination: str | Path,
    intent_filter: str | None = None,
) -> None:
    """Copy training records to a new JSONL file, optionally filtering by intent."""
    records = _read_records(Path(source), intent_filter)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def export_openai(
    source: str | Path,
    destination: str | Path,
    intent_filter: str | None = None,
) -> None:
    """Convert training records to OpenAI fine-tuning chat JSONL format.

    Each record becomes a chat completion example with:
    - system: the intent description
    - user: a generic instruction
    - assistant: the accepted (corrected) output
    """
    records = _read_records(Path(source), intent_filter)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for record in records:
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You must satisfy the following requirement:\n\n"
                            + record["intent_description"]
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Generate a response that satisfies the above requirement.",
                    },
                    {
                        "role": "assistant",
                        "content": record["accepted_output"],
                    },
                ]
            }
            f.write(json.dumps(example) + "\n")
