# Advanced

## Streaming

Validate intent once a streamed LLM response is fully assembled using `StreamCollector`.

### Sync iterator

```python
from semantix import StreamCollector, Intent

class Polite(Intent):
    """The text must be polite and professional."""

collector = StreamCollector(Polite, judge=my_judge)
for chunk in collector.wrap(llm_stream()):
    print(chunk, end="")

result = collector.result()  # Intent instance or raises SemanticIntentError
print(result.text)
```

### Async iterator

```python
async for chunk in collector.awrap(async_llm_stream()):
    print(chunk, end="")

result = collector.result()
```

### Context manager

```python
with StreamCollector(Polite, judge=my_judge) as sc:
    for chunk in llm_stream():
        sc.feed(chunk)

result = sc.result()
```

### Async context manager

```python
async with StreamCollector(Polite, judge=my_judge) as sc:
    async for chunk in async_llm_stream():
        sc.feed(chunk)

result = sc.result()
```

`StreamCollector` accumulates chunks, then validates the complete text when the stream ends or the context manager exits. The `result()` method returns a validated Intent instance or raises `SemanticIntentError`.

## Audit trail

`AuditEngine` produces hash-chained JSON-LD certificates for every validation event. This creates a tamper-evident audit trail.

```python
from semantix.audit.engine import AuditEngine

engine = AuditEngine()

# Record a validation event
engine.record(
    intent="ProfessionalDecline",
    output="Go away!",
    score=0.12,
    passed=False,
    reason="Aggressive language detected",
)

# Verify the chain hasn't been tampered with
assert engine.verify_chain()  # True if all hashes are valid

# Write to disk
from pathlib import Path
engine.flush(Path("audit.jsonl"))
```

### How the chain works

Each certificate contains a `previous_hash` field -- the SHA-256 hash of the previous certificate's JSON. The first entry uses `"GENESIS"` as its previous hash. Modifying any entry breaks the chain from that point forward.

### Certificate structure

```json
{
  "@context": "https://schema.semantix.ai/v1",
  "@type": "SemanticCertificate",
  "id": "urn:semantix:cert:...",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "intent": "ProfessionalDecline",
  "score": 0.12,
  "passed": false,
  "reason": "Aggressive language detected",
  "output_hash": "sha256:...",
  "previous_hash": "sha256:..."
}
```

Note that the output itself is not stored -- only its SHA-256 hash. This preserves privacy while maintaining verifiability.

`AuditEngine` is a thread-safe singleton. All calls to `AuditEngine()` return the same instance.

## MCP server

semantix exposes a Model Context Protocol (MCP) server so any AI agent can validate intents as a tool call.

### Install and run

```bash
pip install "semantix-ai[mcp,nli]"
mcp run semantix/mcp/server.py
```

### Tool: verify_text_intent

The server exposes a single tool:

```
verify_text_intent(text, intent_description, threshold=0.5) -> JSON
```

| Parameter | Description |
|---|---|
| `text` | The text to verify |
| `intent_description` | What the text should convey |
| `threshold` | Minimum entailment score (0-1, default 0.5) |

Returns JSON with `score`, `passed`, `reason`, and `correction_suggestion` on failure. The correction suggestion is a structured Markdown block that the calling agent can use to regenerate the text.

### Example response (failure)

```json
{
  "score": 0.12,
  "passed": false,
  "reason": null,
  "correction_suggestion": "## Semantix Verification Failed\n\n### What went wrong\n- **Score:** 0.1200 (threshold 0.5 not met)\n\n..."
}
```

## Async support

`@validate_intent` transparently supports `async def` functions:

```python
from semantix import Intent, validate_intent

class Polite(Intent):
    """The text must be polite and professional."""

@validate_intent(retries=2)
async def respond(msg: str, semantix_feedback=None) -> Polite:
    prompt = f"Respond to: {msg}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return await call_async_llm(prompt)

# Use with await
result = await respond("handle angry customer")
```

The decorator detects `async def` at decoration time and wraps it in an async wrapper. Self-healing retries, feedback injection, and training collection all work identically in async mode.

## Self-healing feedback

When a decorated function declares `semantix_feedback: Optional[str] = None` in its signature, the decorator automatically injects structured feedback on each retry:

```python
from typing import Optional

@validate_intent(retries=2)
def decline(event: str, semantix_feedback: Optional[str] = None) -> ProfessionalDecline:
    prompt = f"Decline this invite: {event}"
    if semantix_feedback:
        prompt += f"\n\n{semantix_feedback}"
    return call_llm(prompt)
```

The feedback is a Markdown report:

```markdown
## Semantix Self-Healing Feedback

Attempt **1** failed validation.

### What went wrong
- **Intent:** `ProfessionalDecline`
- **Score:** 0.1234 (threshold not met)
- **Judge reason:** Aggressive language detected

### What is required
The text must politely decline an invitation without being rude.

### Your previous output (rejected)
```
Go away, I don't want to come.
```

Please generate a new response that satisfies the requirement above.
```

### Manual feedback via get_last_failure

For custom feedback formatting, use `get_last_failure()`:

```python
from semantix import get_last_failure, validate_intent

@validate_intent(retries=2)
def decline(event: str) -> ProfessionalDecline:
    hint = ""
    if failure := get_last_failure():
        hint = f"\n\nPrevious score: {failure.score:.2f}. Be more polite."
    return call_llm(f"Decline this invite: {event}{hint}")
```

`get_last_failure()` returns the most recent `SemanticIntentError` in the current context (thread/async-safe via `ContextVar`).

## Related

- [Getting Started](getting-started.md) -- basic retries and validation
- [Judges](judges.md) -- choose the right judge for streaming/audit
- [Training](training.md) -- capture correction pairs from retries
