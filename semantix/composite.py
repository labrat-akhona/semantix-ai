"""Composite intent combinators — AllOf / AnyOf."""

from __future__ import annotations

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
    combined_doc = f"ALL of the following requirements must be satisfied:\n\n{descriptions}"
    # Only set threshold explicitly when at least one component does,
    # so the judge's recommended_threshold can apply for default intents.
    explicit = [cls for cls in intent_classes if "threshold" in cls.__dict__]
    ns: dict[str, object] = {
        "__doc__": combined_doc,
        "_component_intents": intent_classes,
        "_compose_kind": "all",
    }
    if explicit:
        ns["threshold"] = min(cls.threshold for cls in explicit)
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
        f"AT LEAST ONE of the following requirements must be satisfied:\n\n{descriptions}"
    )
    # Only set threshold explicitly when at least one component does.
    explicit = [cls for cls in intent_classes if "threshold" in cls.__dict__]
    ns: dict[str, object] = {
        "__doc__": combined_doc,
        "_component_intents": intent_classes,
        "_compose_kind": "any",
    }
    if explicit:
        ns["threshold"] = max(cls.threshold for cls in explicit)
    return type(f"AnyOf[{names}]", (Intent,), ns)


def Not(intent_class: type[Intent]) -> type[Intent]:
    """Create a new Intent that requires the output to **NOT** satisfy the
    given intent.

    Example
    -------
    >>> Safe = Not(MedicalAdvice)
    >>> @validate_intent
    ... def chatbot(msg: str) -> Safe: ...

    Can also use the ``~`` operator::

    >>> Safe = ~MedicalAdvice
    """
    if not isinstance(intent_class, type) or not issubclass(intent_class, Intent):
        raise TypeError(f"Expected an Intent subclass, got {intent_class!r}")
    if intent_class is Intent:
        raise TypeError("Cannot negate the bare Intent base class.")

    original_description = intent_class.description()
    negated_doc = f"The text must NOT satisfy the following:\n\n{original_description}"
    ns: dict[str, object] = {
        "__doc__": negated_doc,
        "_negated_intent": intent_class,
    }
    if "threshold" in intent_class.__dict__:
        ns["threshold"] = intent_class.threshold
    return type(f"Not[{intent_class.__name__}]", (Intent,), ns)


def _validate_intent_classes(classes: tuple[type[Intent], ...]) -> None:
    for cls in classes:
        if not isinstance(cls, type) or not issubclass(cls, Intent):
            raise TypeError(f"Expected an Intent subclass, got {cls!r}")
        if cls is Intent:
            raise TypeError("Cannot combine the bare Intent base class.")
