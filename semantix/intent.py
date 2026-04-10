"""Core Intent base class — the semantic type primitive."""

from __future__ import annotations

from typing import ClassVar


class _IntentMeta(type):
    """Metaclass that enables ``IntentA & IntentB`` and ``IntentA | IntentB``
    syntax for building composite intents."""

    def __and__(cls, other: type) -> type:
        from semantix.composite import AllOf  # noqa: WPS433

        return AllOf(cls, other)  # type: ignore[arg-type]

    def __or__(cls, other: type) -> type:
        from semantix.composite import AnyOf  # noqa: WPS433

        return AnyOf(cls, other)  # type: ignore[arg-type]


class Intent(metaclass=_IntentMeta):
    """Base class for semantic intent types.

    Subclass this and write a docstring that describes the *semantic contract*
    the text must satisfy.  The docstring is extracted at validation time and
    forwarded to a Judge for verification.

    Example
    -------
    >>> class ProfessionalDecline(Intent):
    ...     \"\"\"The text must politely decline an invitation without being
    ...     rude or aggressive.\"\"\"
    """

    # Optional per-subclass override of the similarity threshold (0‒1).
    threshold: ClassVar[float] = 0.8

    def __init__(self, text: str) -> None:
        self._text = text

    # ── public API ──────────────────────────────────────────────

    @property
    def text(self) -> str:
        """The wrapped string value."""
        return self._text

    @classmethod
    def description(cls) -> str:
        """Return the semantic requirement defined in the subclass docstring."""
        doc = cls.__doc__
        if not doc or not doc.strip():
            raise TypeError(
                f"{cls.__name__} must have a non-empty docstring describing the semantic intent."
            )
        return doc.strip()

    # ── dunder helpers ──────────────────────────────────────────

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._text!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Intent):
            return self._text == other._text
        if isinstance(other, str):
            return self._text == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._text)
