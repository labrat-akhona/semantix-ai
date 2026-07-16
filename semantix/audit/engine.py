"""AuditEngine — immutable, hash-chained audit trail for semantic validation.

Every validation event is captured as a JSON-LD Semantic Certificate.
Entries are SHA-256 hash-linked so tampering with any record invalidates
the chain from that point forward.

Schema versions
---------------
Certificates carry an ``@context`` naming their schema version. ``record()``
emits the current version (``…/v2``), which adds first-class ``hypothesis``,
``judge_id``, ``subject``, ``metadata`` and a ``claim_hash`` to the original
(``…/v1``) field set.

The chain is verified by re-hashing each entry's *entire* JSON dict, so it is
agnostic to which fields an entry carries: a ``v1`` certificate written under
the old schema still verifies unchanged, and a chain that upgrades mid-life
(``v1`` rows followed by ``v2`` rows) verifies as one chain. See
``tests/fixtures/v1_chain_frozen.jsonl`` and ``TestBackwardCompatV1Chain``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_CONTEXT_V2 = "https://schema.semantix.ai/v2"


def _claim_hash(premise: str, hypothesis: str | None, judge_id: str | None) -> str | None:
    """SHA-256 over *what was judged*: the (premise, hypothesis, judge_id) triple.

    Returns ``None`` when there is no hypothesis — without a judged claim this
    would add nothing over ``output_hash`` (the premise hash). When a hypothesis
    is present, this distinguishes certificates that judged *different clauses*
    against the *same premise*, which a premise-only ``output_hash`` cannot.

    The inputs are serialized as canonical JSON (``sort_keys=True``) rather than
    concatenated, so no premise/hypothesis text can be crafted to collide with a
    different split of the same bytes.
    """
    if hypothesis is None:
        return None
    payload = json.dumps(
        {"premise": premise, "hypothesis": hypothesis, "judge_id": judge_id},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ChainReport:
    """Summary of an audit chain — integrity *and* variety.

    ``verify_chain()`` proves a chain is intact. It does not prove the chain
    certifies more than one thing: a chain can verify perfectly while every
    certificate carries the same verdict. This report surfaces that so a
    constant chain is visible without a human having to notice.

    Attributes
    ----------
    valid:
        Result of hash-chain verification.
    n_certs:
        Number of certificates.
    distinct_verdicts:
        Distinct ``(passed, score)`` outcomes.
    distinct_claims:
        Distinct judged claims — ``claim_hash`` where present, falling back to
        ``output_hash`` for ``v1`` certificates that predate ``claim_hash``.
    distinct_premises:
        Distinct ``output_hash`` (premise) values.
    is_constant:
        ``True`` when there is more than one certificate but only one distinct
        verdict — the chain certifies one repeated result.
    """

    valid: bool
    n_certs: int
    distinct_verdicts: int
    distinct_claims: int
    distinct_premises: int
    is_constant: bool

    def __str__(self) -> str:
        head = "valid" if self.valid else "BROKEN"
        base = (
            f"ChainReport({head}, {self.n_certs} certs, "
            f"{self.distinct_verdicts} distinct verdict(s), "
            f"{self.distinct_claims} distinct claim(s), "
            f"{self.distinct_premises} distinct premise(s))"
        )
        if self.is_constant:
            base += (
                " -- CONSTANT: every certificate carries the same verdict; the "
                "chain is intact but certifies one repeated result."
            )
        return base


class AuditEngine:
    """Thread-safe singleton that captures validation events as hash-chained
    JSON-LD certificates.

    Usage
    -----
    >>> engine = AuditEngine()
    >>> engine.record(intent="PoliteDecline", output="Go away", score=0.2, passed=False)
    >>> engine.verify_chain()  # True if no tampering
    >>> engine.chain_report()  # integrity + variety (catches a constant chain)
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
        hypothesis: str | None = None,
        judge_id: str | None = None,
        subject: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Append a new Semantic Certificate to the audit trail. Returns the certificate dict.

        Parameters
        ----------
        intent:
            The Intent / requirement label the output was validated against.
        output:
            The premise — the text that was judged. Stored only as
            ``output_hash`` (never in the clear).
        score, passed, reason:
            The verdict and its explanation.
        hypothesis:
            *What was judged* — the specific claim/clause the premise was scored
            against (e.g. a POPIA §72 statement). First-class as of schema v2;
            previously callers had to smuggle this into ``intent``.
        judge_id:
            *By which judge* — an identifier for the judge, version and
            configuration (e.g. ``"POPIAJudge/v1@0.75"``).
        subject:
            *About whom/what* — the data subject or application the certificate
            concerns (e.g. ``"user:5b4c9d12"``).
        metadata:
            Any additional structured context (destination, country, corpus
            version, …). Must be JSON-serializable.

        The new fields are keyword-only with ``None`` defaults, so code written
        against the 0.2.3 signature keeps working. New certificates emit the v2
        ``@context``; existing v1 certificates on disk are unaffected and still
        verify.
        """
        with self._lock:
            previous_hash = (
                "GENESIS"
                if not self._entries
                else hashlib.sha256(
                    json.dumps(self._entries[-1], sort_keys=True).encode()
                ).hexdigest()
            )

            cert = {
                "@context": _CONTEXT_V2,
                "@type": "SemanticCertificate",
                "id": f"urn:semantix:cert:{uuid.uuid4()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "intent": intent,
                "hypothesis": hypothesis,
                "subject": subject,
                "judge_id": judge_id,
                "metadata": metadata,
                "score": score,
                "passed": passed,
                "reason": reason,
                "output_hash": hashlib.sha256(output.encode()).hexdigest(),
                "claim_hash": _claim_hash(output, hypothesis, judge_id),
                "previous_hash": previous_hash,
            }
            self._entries.append(cert)
            return cert

    @staticmethod
    def verify_entries(entries: list[dict]) -> bool:
        """Verify integrity of an arbitrary list of certificates. Returns True if valid.

        Field-set agnostic: it re-hashes each entry's whole JSON dict, so v1 and
        v2 certificates (and mixed chains) verify the same way. Use this to
        check a chain loaded from disk without mutating engine state.
        """
        for i, entry in enumerate(entries):
            if i == 0:
                if entry.get("previous_hash") != "GENESIS":
                    return False
            else:
                expected = hashlib.sha256(
                    json.dumps(entries[i - 1], sort_keys=True).encode()
                ).hexdigest()
                if entry.get("previous_hash") != expected:
                    return False
        return True

    def verify_chain(self) -> bool:
        """Verify integrity of the entire in-memory audit trail. Returns True if valid."""
        return self.verify_entries(self._entries)

    @staticmethod
    def summarize(entries: list[dict]) -> ChainReport:
        """Build a :class:`ChainReport` (integrity + variety) for *entries*."""
        n = len(entries)
        verdicts = {(e.get("passed"), e.get("score")) for e in entries}
        claims = {(e.get("claim_hash") or e.get("output_hash")) for e in entries}
        premises = {e.get("output_hash") for e in entries}
        return ChainReport(
            valid=AuditEngine.verify_entries(entries),
            n_certs=n,
            distinct_verdicts=len(verdicts),
            distinct_claims=len(claims),
            distinct_premises=len(premises),
            is_constant=n > 1 and len(verdicts) == 1,
        )

    def chain_report(self) -> ChainReport:
        """Summarize the in-memory chain — integrity *and* variety.

        A chain that verifies perfectly can still certify one repeated fact.
        ``chain_report().is_constant`` catches exactly that.
        """
        return self.summarize(self._entries)

    def load(self, path: Path) -> AuditEngine:
        """Load an existing JSONL chain into this engine, replacing current entries.

        Models resuming an on-disk chain (to append more certificates) or
        verifying one produced by an earlier run. Returns ``self`` for chaining.
        """
        with self._lock:
            loaded: list[dict] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        loaded.append(json.loads(line))
            self._entries.clear()
            self._entries.extend(loaded)
        return self

    def flush(self, path: Path) -> None:
        """Write all entries to a JSONL file."""
        with self._lock, open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
