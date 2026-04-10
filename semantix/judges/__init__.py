"""Abstract Judge interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Verdict:
    """Result returned by every Judge implementation.

    Attributes
    ----------
    passed : bool
        ``True`` if the output satisfies the intent.
    score : float | None
        A 0–1 confidence / similarity score when available.
    reason : str | None
        Optional explanation (populated by LLM-based judges).
    """

    passed: bool
    score: float | None = None
    reason: str | None = None


class Judge(ABC):
    """Base interface that all judge backends must implement."""

    recommended_threshold: ClassVar[float | None] = None

    @abstractmethod
    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        """Decide whether *output* satisfies the *intent_description*.

        Parameters
        ----------
        output:
            The text to validate.
        intent_description:
            The semantic requirement (from the Intent docstring).
        threshold:
            Minimum acceptable score / confidence.

        Returns
        -------
        Verdict
        """
        ...
