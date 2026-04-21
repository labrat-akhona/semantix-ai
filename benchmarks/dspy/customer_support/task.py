"""DSPy program + dataset loader for customer-support QA."""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from benchmarks.common.runner import Example

DATASET = Path(__file__).parent / "dataset.json"


class CustomerSupportResponse(dspy.Signature):
    """Generate a professional customer-service response to an incoming message."""

    customer_message: str = dspy.InputField()
    response: str = dspy.OutputField()


def load_examples() -> list[Example]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    return [
        Example(
            example_id=d["example_id"],
            text="",  # Filled in after DSPy generation
            intent=d["intent_description"],
            input={"customer_message": d["customer_message"]},
        )
        for d in data
    ]


def make_program() -> dspy.Module:
    return dspy.ChainOfThought(CustomerSupportResponse)


def generate_all(examples: list[Example], program: dspy.Module) -> list[Example]:
    """Run the DSPy program once per example, populating Example.text."""
    out: list[Example] = []
    for ex in examples:
        try:
            pred = program(**(ex.input or {}))
            text = pred.response
        except Exception as exc:  # noqa: BLE001
            text = f"__GENERATION_ERROR__:{type(exc).__name__}:{exc}"
        out.append(
            Example(example_id=ex.example_id, text=text, intent=ex.intent, input=ex.input)
        )
    return out
