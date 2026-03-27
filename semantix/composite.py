"""Composite intent combinators — AllOf / AnyOf."""

from __future__ import annotations

from typing import ClassVar

from semantix.intent import Intent


def AllOf(*intent_classes: type[Intent]) -> type[Intent]:
    """Create a new Intent that requires the output to satisfy **all** of the
    given intents.

    Example
    -------
    >>> Polite = AllOf(ProfessionalDecline, PositiveSentiment)
    >>> @validate_intent
    ... def respond(msg: str) -> Polite: ...
    """
    if len(intent_classes) < 2:
        raise TypeError("AllOf requires at least two Intent subclasses.")
    _validate_intent_classes(intent_classes)

    names = " & ".join(cls.__name__ for cls in intent_classes)
    descriptions = "\n\nAND\n\n".join(cls.description() for cls in intent_classes)
    combined_doc = (
        f"ALL of the following requirements must be satisfied:\n\n"
        f"{descriptions}"
    )
    # Use the minimum threshold across the constituents.
    min_threshold: float = min(cls.threshold for cls in intent_classes)

    ns: dict[str, object] = {
        "__doc__": combined_doc,
        "threshold": min_threshold,
        "_component_intents": intent_classes,
    }
    return type(f"AllOf[{names}]", (Intent,), ns)


def AnyOf(*intent_classes: type[Intent]) -> type[Intent]:
    """Create a new Intent that requires the output to satisfy **at least one**
    of the given intents.

    Example
    -------
    >>> Flexible = AnyOf(ProfessionalDecline, CasualDecline)
    >>> @validate_intent
    ... def respond(msg: str) -> Flexible: ...
    """
    if len(intent_classes) < 2:
        raise TypeError("AnyOf requires at least two Intent subclasses.")
    _validate_intent_classes(intent_classes)

    names = " | ".join(cls.__name__ for cls in intent_classes)
    descriptions = "\n\nOR\n\n".join(cls.description() for cls in intent_classes)
    combined_doc = (
        f"AT LEAST ONE of the following requirements must be satisfied:\n\n"
        f"{descriptions}"
    )
    max_threshold: float = max(cls.threshold for cls in intent_classes)

    ns: dict[str, object] = {
        "__doc__": combined_doc,
        "threshold": max_threshold,
        "_component_intents": intent_classes,
    }
    return type(f"AnyOf[{names}]", (Intent,), ns)


def _validate_intent_classes(classes: tuple[type[Intent], ...]) -> None:
    for cls in classes:
        if not isinstance(cls, type) or not issubclass(cls, Intent):
            raise TypeError(
                f"Expected an Intent subclass, got {cls!r}"
            )
        if cls is Intent:
            raise TypeError("Cannot combine the bare Intent base class.")
