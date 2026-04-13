# Composition

semantix lets you combine intents using logical operators. This is how you build complex semantic contracts from simple building blocks.

## Not -- negation

Block what your model must NOT say:

```python
from semantix import Intent, Not

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

Safe = Not(MedicalAdvice)
# or use the ~ operator:
Safe = ~MedicalAdvice
```

`Not` creates a new Intent whose description is: *"The text must NOT satisfy the following: ..."*

Use this for compliance -- blocking PII, medical advice, legal advice, competitor mentions, etc.

```python
@validate_intent
def chatbot(msg: str) -> Safe:
    return call_my_llm(msg)
```

## AllOf -- all must pass

Require the output to satisfy **every** intent:

```python
from semantix import Intent, AllOf

class Polite(Intent):
    """The text must be polite and professional."""

class Helpful(Intent):
    """The text must provide a useful and actionable answer."""

PoliteAndHelpful = AllOf(Polite, Helpful)
```

Or use the `&` operator:

```python
PoliteAndHelpful = Polite & Helpful
```

The combined description joins all component descriptions with "AND". The threshold is the **minimum** across all components that explicitly set one.

## AnyOf -- at least one must pass

Require the output to satisfy **at least one** intent:

```python
from semantix import Intent, AnyOf

class FormalDecline(Intent):
    """The text formally declines the request with business language."""

class CasualDecline(Intent):
    """The text casually declines the request in a friendly tone."""

FlexibleDecline = AnyOf(FormalDecline, CasualDecline)
```

Or use the `|` operator:

```python
FlexibleDecline = FormalDecline | CasualDecline
```

The combined description joins all component descriptions with "OR". The threshold is the **maximum** across all components that explicitly set one.

## Combining operators

Build complex safety contracts by chaining operators:

```python
class Polite(Intent):
    """The text must be polite and professional."""

class MedicalAdvice(Intent):
    """The text provides medical diagnoses or treatment recommendations."""

class LegalAdvice(Intent):
    """The text provides legal counsel or recommendations."""

SafeAndPolite = Polite & ~MedicalAdvice & ~LegalAdvice
```

```python
@validate_intent
def customer_support(msg: str) -> SafeAndPolite:
    return call_my_llm(msg)
```

## How it works

Under the hood, `AllOf`, `AnyOf`, and `Not` create new Intent subclasses dynamically. The combined docstring is what gets passed to the judge:

```python
print(SafeAndPolite.description())
# ALL of the following requirements must be satisfied:
#
# The text must be polite and professional.
#
# AND
#
# The text must NOT satisfy the following:
#
# The text provides medical diagnoses or treatment recommendations.
#
# AND
#
# The text must NOT satisfy the following:
#
# The text provides legal counsel or recommendations.
```

## Rules

- `AllOf` and `AnyOf` require **at least two** Intent subclasses.
- You cannot combine the bare `Intent` base class -- only subclasses with docstrings.
- `Not` requires a single Intent subclass (not the bare `Intent` base).

## Related

- [Intents](intents.md) -- defining semantic contracts
- [Judges](../judges.md) -- how the combined description is evaluated
