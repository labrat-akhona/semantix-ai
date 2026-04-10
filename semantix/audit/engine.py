"""AuditEngine — immutable, hash-chained audit trail for semantic validation.

Every validation event is captured as a JSON-LD Semantic Certificate.
Entries are SHA-256 hash-linked so tampering with any record invalidates
the chain from that point forward.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AuditEngine:
    """Thread-safe singleton that captures validation events as hash-chained
    JSON-LD certificates.

    Usage
    -----
    >>> engine = AuditEngine()
    >>> engine.record(intent="PoliteDecline", output="Go away", score=0.2, passed=False)
    >>> engine.verify_chain()  # True if no tampering
    >>> engine.flush(Path("audit.jsonl"))
    """

    _instance: AuditEngine | None = None
    _entries: list[dict] = []
    _lock: threading.Lock | None = None

    def __new__(cls) -> AuditEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._entries = []
            cls._lock = threading.Lock()
        return cls._instance

    @property
    def entries(self) -> list[dict]:
        return self._entries

    def record(
        self,
        *,
        intent: str,
        output: str,
        score: float,
        passed: bool,
        reason: str | None = None,
    ) -> dict:
        """Append a new Semantic Certificate to the audit trail. Returns the certificate dict."""
        with self._lock:
            previous_hash = (
                "GENESIS"
                if not self._entries
                else hashlib.sha256(
                    json.dumps(self._entries[-1], sort_keys=True).encode()
                ).hexdigest()
            )

            cert = {
                "@context": "https://schema.semantix.ai/v1",
                "@type": "SemanticCertificate",
                "id": f"urn:semantix:cert:{uuid.uuid4()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "intent": intent,
                "score": score,
                "passed": passed,
                "reason": reason,
                "output_hash": hashlib.sha256(output.encode()).hexdigest(),
                "previous_hash": previous_hash,
            }
            self._entries.append(cert)
            return cert

    def verify_chain(self) -> bool:
        """Verify integrity of the entire audit trail. Returns True if valid."""
        for i, entry in enumerate(self._entries):
            if i == 0:
                if entry["previous_hash"] != "GENESIS":
                    return False
            else:
                expected = hashlib.sha256(
                    json.dumps(self._entries[i - 1], sort_keys=True).encode()
                ).hexdigest()
                if entry["previous_hash"] != expected:
                    return False
        return True

    def flush(self, path: Path) -> None:
        """Write all entries to a JSONL file."""
        with self._lock, open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
