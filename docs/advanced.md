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

Each explicit `AuditEngine.record()` call produces a hash-chained JSON-LD certificate.
The decorator does not write certificates automatically. Record the actual verdict
and the claim and judge that produced it, then call `flush()` to persist the chain.

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

Each certificate contains a `previous_hash` field -- the SHA-256 hash of the previous
certificate's JSON (`json.dumps(entry, sort_keys=True)`, UTF-8). The first entry uses
`"GENESIS"`. Modifying an entry breaks its successor's link.

Verification establishes internal link consistency only. It cannot detect an edited
final entry, removal of trailing entries, or an attacker rewriting the entire chain.
Keep a trusted external checkpoint of the final certificate hash and entry count when
you need to detect those changes. Certificates are not cryptographically signed.

### Certificate structure

```json
{
  "@context": "https://schema.semantix.ai/v2",
  "@type": "SemanticCertificate",
  "id": "urn:semantix:cert:...",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "intent": "ProfessionalDecline",
  "hypothesis": "The text politely declines an invitation.",
  "judge_id": "QuantizedNLIJudge",
  "subject": null,
  "metadata": null,
  "score": 0.12,
  "passed": false,
  "reason": "Aggressive language detected",
  "output_hash": "<64-character SHA-256 hex digest>",
  "claim_hash": "<64-character SHA-256 hex digest>",
  "previous_hash": "GENESIS"
}
```

The raw output is not stored in `output_hash`. Other fields, including `reason`,
`hypothesis`, `subject`, and `metadata`, are stored as supplied and can contain personal
information. Hashing alone does not anonymize guessable input. v1 and mixed v1/v2
chains remain supported without rewriting their certificates.

`AuditEngine` is a thread-safe singleton. All calls to `AuditEngine()` return the same instance.

Call `AuditEngine.reset()` between independent audits; existing references then see
the fresh chain. `engine.load(path)` resumes an existing UTF-8 JSONL file and rejects
malformed rows or broken links before replacing in-memory entries. To inspect a
suspect file without loading it for appending, run `semantix verify audit.jsonl`.
The verifier returns 0 for consistent links (including an empty file), 1 for a broken
chain, and 2 for unreadable or malformed input. It also reports repeated verdicts.

## MCP server

semantix exposes a Model Context Protocol (MCP) server so any AI agent can validate intents as a tool call.

This integration uses the v1 SDK's `FastMCP` API. The `mcp` extra constrains the SDK
to `>=1.0,<2`; upgrading to the v2 `MCPServer` API requires a separate integration
migration. See the [official migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver).

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
