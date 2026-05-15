# TrustMesh integration feedback — semantix-ai 0.2.0

*Filed 2026-05-15 from feedback compiled by the TrustMesh team during
Phase 2.5 integration on 2026-04-28. Real-world friction; the suggested
fixes are sketches, not finished patches. Triage and convert to issues
as we get to each one.*

---

## 1. Critical — `validate_intent` silently no-ops on unresolvable annotations

**Where:** `semantix/decorator.py::_resolve_intent_class`

**Symptom:** A decorated function passes through every output unchecked, with no
warning, no log line, no exception. Took ~30 minutes to diagnose.

**Cause:** `_resolve_intent_class` calls `get_type_hints(func)` inside a
bare `try / except: return None`. If the function has any annotation that
can't be resolved at runtime — a forward reference, a `TYPE_CHECKING`-only
import, a string annotation referencing a symbol from another module — the
NameError is swallowed and the decorator becomes a silent no-op.

**Reproducer:**

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from semantix import Intent, validate_intent

if TYPE_CHECKING:
    from some_module import SomeOtherType  # not imported at runtime

class Polite(Intent):
    """The text is polite."""

@validate_intent(retries=2)
async def reply(ctx: SomeOtherType) -> Polite:  # SomeOtherType unresolvable
    return "🤬"  # validation never runs — bad output sails through
```

**Suggested fix:**

```python
def _resolve_intent_class(func):
    try:
        hints = get_type_hints(func)
    except Exception as exc:
        logger.warning(
            "validate_intent: cannot resolve annotations for %s (%s) — "
            "decorator will no-op. Check for forward references or "
            "TYPE_CHECKING-only imports in the function signature.",
            getattr(func, "__qualname__", func), exc,
        )
        return None
    ...
```

A warning is enough — failing loudly might break callers, but the silent
case has no signal at all.

---

## 2. Critical — composite Intent descriptions don't entail well in NLI judges

**Where:** Composite intents built with `&` / `|` / `~`.

**Symptom:** `@validate_intent(A & ~B & ~C)` rejects even ideal output.

**Cause:** The composite's `description()` produces text like:

> "ALL of the following requirements must be satisfied:
> ALL of the following requirements must be satisfied:
> The text must NOT satisfy the following: Text gives advice about money…
> AND The text must NOT satisfy the following: …"

Cross-encoder NLI judges (`QuantizedNLIJudge`, `NLIJudge`) treat this as
one premise/hypothesis pair and try to entail the whole multi-clause
nested description. Empirically, even ideal output scores ~0.02 against
this — well below any reasonable threshold.

Per-leaf evaluation works fine: each leaf description is one short
sentence the NLI model can directly entail or rule out.

**Calibration data from TrustMesh:**

| Intent (single, ~10 words) | Good text | Bad text matching this intent |
|---|---|---|
| "Text gives advice about money…" | 0.04 | 0.72 ✓ |
| "Text uses emoji or internet slang" | 0.07 | 0.83 ✓ |
| Composite of three above | 0.02 | 0.06 |

**Suggested fixes (any of these would help, in order of preference):**

1. **Per-leaf evaluation in the composite.** When the composite is
   evaluated, decompose into leaves and AND/OR/NOT the verdicts. Score
   becomes the min/max/inverted-min as appropriate. This matches what a
   reasonable user expects "all of these constraints" to mean.
2. **Document the limitation.** Add a section in the README warning that
   composites work well with `LLMJudge` / `EmbeddingJudge` but NLI judges
   benefit from per-leaf evaluation, and show a workaround.
3. **Add a `judge="leaf"` flag** on the decorator that opts into per-leaf
   semantics for users who want it.

This was the highest-impact correction we had to make; we ended up
abandoning the composite decorator entirely and writing a 12-line manual
loop against the leaf intents.

---

## 3. Medium — README example `@validate_intent(~MedicalAdvice)` doesn't work

**Where:** README "What you get → 1. Validation as a decorator"

**Symptom:** The example

```python
@validate_intent(~MedicalAdvice)
def chatbot(msg: str) -> str:
    return call_my_llm(msg)
```

is a no-op. The decorator only looks at the function's return
**annotation**, not the positional argument. The actual signature is
`validate_intent(func=None, *, judge=None, retries=0, collector=None)` —
when you pass an Intent class as the first arg, the decorator code calls
`decorator(intent_class)` which then `_resolve_intent_class(intent_class)`
returns `None` (Intent classes don't have a `return` annotation), so the
decorator gives back the class unchanged and the underlying function is
never wrapped. Combined with finding #1, the symptom is total silence.

**Suggested fix:** make the first positional argument honour an Intent
subclass, OR remove that example from the README and only show the
return-annotation pattern (which works correctly).

---

## 4. Medium — default threshold inconsistency

**Where:** `Judge.evaluate(threshold=0.5)` vs `_run_judge` inside the decorator.

* Direct call: `judge.evaluate(text, desc)` defaults to `threshold=0.5`.
* Through the decorator: `_run_judge` uses `judge.recommended_threshold`
  (0.3 for `QuantizedNLIJudge`).

This gives different verdicts for the same `(text, intent)` pair depending
on whether you call the judge directly or via the decorator. Surprising.

**Suggested fix:** either document this clearly, or have
`Judge.evaluate`'s default fall through to `self.recommended_threshold`
when no explicit threshold is passed. The latter is what most users
probably already assume is happening.

---

## 5. Medium — `recommended_threshold` is a property, not a method

**Where:** `QuantizedNLIJudge.recommended_threshold`

```python
>>> j = QuantizedNLIJudge()
>>> j.recommended_threshold()
TypeError: 'float' object is not callable
```

There's no docstring on the attribute, and a reasonable user familiar with
Java/`get_X` patterns will reach for parentheses first. Either:

* Document it as a class attribute / property in the README, or
* Make it callable as `j.recommended_threshold()` for symmetry with
  configurable getters elsewhere in the codebase.

---

## 6. Medium — async support is undocumented in the README

**Where:** README examples; `decorator.py::async_wrapper`.

`@validate_intent` correctly handles `async def` functions
(`async_wrapper` exists in `decorator.py`), but every example in the
README uses `def`. Users building async-first apps (FastAPI, asyncio
agents) reasonably assume they need to do something extra to opt in.

**Suggested fix:** add a one-paragraph "Async" section to the README
showing the same pattern with `async def` + `await`.

---

## 7. Low — undocumented return type from validated functions

**Where:** Behaviour of `@validate_intent` on success.

The README says "Returns the validated reply" and uses the result like a
string in subsequent code. In practice the decorator returns
`intent_cls(raw_output)` — an instance of the Intent subclass. This works
for string-like uses (so subclasses presumably implement `__str__` or
extend `str`), but it's surprising:

* `type(reply) is str` → False
* `isinstance(reply, str)` → True or False depending on Intent's MRO
* JSON-serializing `reply` may behave unexpectedly

**Suggested fix:** README should state explicitly that the return value
is an `Intent` instance that is string-compatible, and ideally show
`json.dumps(reply)` working as expected (or call out that you must call
`str(reply)` first for serialization).

---

## 8. Low — AuditEngine isn't auto-wired to `validate_intent`

**Where:** Decorator vs `AuditEngine`.

The decorator validates but does not write to the AuditEngine. To get a
hash-chained certificate per call, you must catch `SemanticIntentError` /
intercept the success path and call `AuditEngine().record(...)` yourself.
Most users want both behaviours together — decorating a function to "get
audit + retries + validation" is a natural ask.

**Suggested fix:** add an opt-in parameter:

```python
@validate_intent(judge=NLIJudge(), retries=2, audit=True)
def reply(...) -> ResolutionPolite: ...
```

`audit=True` would write to `AuditEngine()` (singleton) on both pass
and fail. Could also accept `audit=AuditEngine(...)` for explicit
injection.

(The Phase 2.5 brief I was given had `audit=True` in the example code,
which suggests the intended/expected API surface.)

---

## 9. Low — `AuditEngine.flush(path)` is the only persistence path

**Where:** `semantix.audit.engine.AuditEngine`

The README says "every validation produces a signed JSON-LD certificate
hash-chained to the previous one" but doesn't explain that those
certificates live entirely in memory until you call `engine.flush(path)`.
A long-running service that crashes loses its entire audit chain unless
the operator wired `flush()` somewhere.

**Suggested fix(es):**

* Document the in-memory + flush model in the README's audit section.
* Optionally accept `AuditEngine(path="…", autoflush=True)` so each
  `record()` call appends-and-fsyncs to the JSONL file.

---

## 10. Bonus — package surface that isn't in the README

`semantix.POPIAJudge` is exported but not mentioned in the README. For
South-African / POPIA-specific use cases (which is exactly what TrustMesh
is) this would be much easier to discover with a one-line callout in the
"Pluggable judges" section.

---

## What worked well

To balance the criticism: the install was clean (`pip install
semantix-ai[turbo]` Just Worked, ONNX runtime resolved, tokenizers
auto-downloaded the quantized model), latency was ~15-50ms per check
exactly as advertised, the audit chain hash-tampering test passed on
first attempt, the per-leaf evaluation we ended up with is genuinely
useful, and `Verdict.score` / `SemanticIntentError.score` carry the
exact right amount of structured info to make graceful-degradation easy
to wire up.

---

## Triage at intake (2026-05-15)

* **Criticals (#1, #2)** are correctness bugs — silent no-op is the
  worst failure mode for a guardrail library, and composite-intent
  scoring being broken with NLI judges defeats the documented
  composability story. These should move to the top of the next
  development sprint once distribution work clears.
* **Mediums (#3–#6)** are mostly README/docs corrections plus one API
  consistency call (#4). Cheap to fix in a single docs PR.
* **Lows (#7–#10)** are nice-to-haves; defer until #1, #2 are closed.

Priority order suggested at intake: #1 → #2 → #3 (since #3 compounds
with #1) → #4 → docs cluster (#6, #7, #9, #10) → #8 → #5.
