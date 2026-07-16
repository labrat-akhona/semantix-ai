# Frozen test fixtures — do not regenerate

## `v1_chain_frozen.jsonl`

A 5-entry hash-chained audit trail emitted by the **v0.2.3** `AuditEngine.record()`
(the `@context: https://schema.semantix.ai/v1` certificate format shipped on PyPI).

**This file is an immutability witness. Never regenerate or edit it.**

The whole value proposition of the audit certificate is that a chain written
yesterday still verifies tomorrow — the math proves the chain, not the database.
`test_audit_engine.py::TestBackwardCompatV1Chain` loads these exact bytes and
asserts `verify_entries()` still returns `True`. If a future schema change
(v2 `@context`, new certificate fields, a different `claim_hash`) alters how
existing entries are hashed, this test fails loudly — which is the point.

If you think you need to change this file, you are about to break every v1
certificate any adopter has ever written. Add a new fixture instead.

Provenance: generated 2026-07-16 from the unmodified 0.2.3 `engine.py`, before
the v0.3.0 first-class-fields / `claim_hash` work.
