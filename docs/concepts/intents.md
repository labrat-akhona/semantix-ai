# Intents

An **Intent** is a class whose docstring defines the semantic contract that text must satisfy. It is the core primitive of semantix.

## Defining an Intent

```python
from semantix import Intent

class ProfessionalDecline(Intent):
    """The text must politely decline an invitation without being rude or aggressive."""
```

The docstring is the requirement. At validation time, semantix extracts it and forwards it to a [Judge](../judges.md) that checks whether the output *entails* the description.

!!! warning
    An Intent **must** have a non-empty docstring. Defining one without a docstring raises `TypeError` at validation time.

## Using Intents with `@validate_intent`

```python
from semantix import validate_intent

@validate_intent
def decline_invite(event: str) -> ProfessionalDecline:
    return call_my_llm(event)

result = decline_invite("the company retreat")
```

The decorator inspects the return type annotation. If it's an Intent subclass, it validates the raw string output against the docstring. On success, it wraps the string in an instance of the Intent class.

## The Intent instance

An Intent instance wraps a validated string:

```python
result = decline_invite("the retreat")
print(result.text)    # the raw string
print(str(result))    # same as .text
print(repr(result))   # ProfessionalDecline('...')
```

Intents support equality comparison with other Intents and with plain strings:

```python
result == "some text"        # compares .text
result == other_intent       # compares .text values
```

## Threshold

Each Intent has a `threshold` class variable (default `0.8`). This is the minimum score a judge must return for the output to pass.

```python
class StrictPolite(Intent):
    """The text must be polite and professional."""
    threshold = 0.95  # require very high confidence
```

When no explicit threshold is set in the Intent's `__dict__`, the judge's `recommended_threshold` is used instead. This means the default `0.8` from the base class is effectively overridden by each judge's calibrated default:

| Judge | Recommended threshold |
|---|---|
| NLIJudge | 0.3 |
| QuantizedNLIJudge | 0.3 |
| EmbeddingJudge | 0.8 |
| LLMJudge | 0.7 |

!!! tip
    You usually don't need to set `threshold` manually. The judge's recommended threshold works well for most cases. Only override it when you need stricter or more lenient validation for a specific intent.

## The `description()` method

Every Intent exposes its docstring via the `description()` class method:

```python
ProfessionalDecline.description()
# "The text must politely decline an invitation without being rude or aggressive."
```

This is what gets passed to the judge. You can also use `description()` in your own code for logging, debugging, or custom validation flows.

## Dynamic Intents

When using [`assert_semantic()`](../testing/assert-semantic.md), you can pass a plain string instead of an Intent class. Internally, semantix creates a dynamic Intent subclass with that string as the docstring:

```python
from semantix.testing import assert_semantic

assert_semantic(output, "polite and professional")
# Equivalent to defining a class with docstring "polite and professional"
```

## Negation

Use `~` or `Not()` to invert an intent -- the output must NOT satisfy the description:

```python
from semantix import Not

Safe = ~MedicalAdvice       # operator syntax
Safe = Not(MedicalAdvice)   # function syntax
```

See [Composition](composition.md) for the full details.

## Related

- [Composition](composition.md) -- combine intents with `Not`, `AllOf`, `AnyOf`
- [Judges](../judges.md) -- how outputs are evaluated against intents
- [Testing](../testing/assert-semantic.md) -- use intents in test assertions
