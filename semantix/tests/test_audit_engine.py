"""Tests for AuditEngine — singleton, hash chain, JSON-LD schema."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from semantix.audit.engine import AuditEngine


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the singleton between tests."""
    AuditEngine._instance = None
    AuditEngine._entries = []
    AuditEngine._lock = None
    yield
    AuditEngine._instance = None
    AuditEngine._entries = []
    AuditEngine._lock = None


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance(self):
        a = AuditEngine()
        b = AuditEngine()
        assert a is b

    def test_shared_state(self):
        a = AuditEngine()
        a.record(intent="TestIntent", output="hello", score=0.9, passed=True)
        b = AuditEngine()
        assert len(b.entries) == 1


# ---------------------------------------------------------------------------
# JSON-LD schema
# ---------------------------------------------------------------------------


class TestCertificateSchema:
    def test_has_context(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert "@context" in entry

    def test_has_type(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert entry["@type"] == "SemanticCertificate"

    def test_has_id(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert "id" in entry
        assert entry["id"].startswith("urn:semantix:cert:")

    def test_has_timestamp(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert "timestamp" in entry

    def test_has_output_hash_not_raw_text(self):
        engine = AuditEngine()
        engine.record(intent="X", output="secret text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert "output_hash" in entry
        assert "secret text" not in json.dumps(entry)

    def test_output_hash_is_sha256(self):
        engine = AuditEngine()
        engine.record(intent="X", output="hello world", score=0.5, passed=True)
        entry = engine.entries[0]
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert entry["output_hash"] == expected

    def test_has_score_and_passed(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.85, passed=True)
        entry = engine.entries[0]
        assert entry["score"] == 0.85
        assert entry["passed"] is True

    def test_has_reason_field(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=False, reason="too vague")
        entry = engine.entries[0]
        assert entry["reason"] == "too vague"

    def test_reason_defaults_to_none(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        entry = engine.entries[0]
        assert entry["reason"] is None

    def test_serializable_to_json(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        json.dumps(engine.entries[0])  # must not raise


# ---------------------------------------------------------------------------
# Hash chain — tamper evidence
# ---------------------------------------------------------------------------


class TestHashChain:
    def test_genesis_entry_has_genesis_previous(self):
        engine = AuditEngine()
        engine.record(intent="X", output="a", score=0.5, passed=True)
        assert engine.entries[0]["previous_hash"] == "GENESIS"

    def test_second_entry_links_to_first(self):
        engine = AuditEngine()
        engine.record(intent="X", output="a", score=0.5, passed=True)
        engine.record(intent="Y", output="b", score=0.6, passed=True)
        first_hash = hashlib.sha256(
            json.dumps(engine.entries[0], sort_keys=True).encode()
        ).hexdigest()
        assert engine.entries[1]["previous_hash"] == first_hash

    def test_chain_of_three(self):
        engine = AuditEngine()
        for i in range(3):
            engine.record(intent=f"I{i}", output=f"t{i}", score=0.5, passed=True)
        for i in range(1, 3):
            prev_hash = hashlib.sha256(
                json.dumps(engine.entries[i - 1], sort_keys=True).encode()
            ).hexdigest()
            assert engine.entries[i]["previous_hash"] == prev_hash

    def test_verify_chain_integrity(self):
        engine = AuditEngine()
        for i in range(5):
            engine.record(intent=f"I{i}", output=f"t{i}", score=0.5, passed=True)
        assert engine.verify_chain() is True

    def test_tampering_detected(self):
        engine = AuditEngine()
        for i in range(3):
            engine.record(intent=f"I{i}", output=f"t{i}", score=0.5, passed=True)
        engine.entries[0]["score"] = 9.99
        assert engine.verify_chain() is False


# ---------------------------------------------------------------------------
# Flush to disk
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_creates_file(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            engine.flush(path)
            assert path.exists()

    def test_flush_writes_valid_jsonl(self):
        engine = AuditEngine()
        engine.record(intent="X", output="a", score=0.5, passed=True)
        engine.record(intent="Y", output="b", score=0.6, passed=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            engine.flush(path)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 2
            for line in lines:
                json.loads(line)  # must not raise

    def test_flush_preserves_entries_in_memory(self):
        engine = AuditEngine()
        engine.record(intent="X", output="text", score=0.5, passed=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.flush(Path(tmpdir) / "audit.jsonl")
        assert len(engine.entries) == 1


# ---------------------------------------------------------------------------
# F2 — first-class certificate fields (bind a cert to a claim, a judge, a subject)
#
# record(*, intent, output, score, passed, reason) had no parameter for WHAT
# was judged (the hypothesis/clause), BY WHICH judge, or ABOUT WHOM. Callers had
# to smuggle all of it into the free-text `intent` string. New certs are v2 and
# carry these as first-class, keyword-only, defaulted fields.
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"


class TestFirstClassFields:
    def test_records_hypothesis(self):
        engine = AuditEngine()
        cert = engine.record(
            intent="POPIA §72",
            output="policy text",
            score=0.5,
            passed=False,
            hypothesis="Personal information is transferred outside South Africa",
        )
        assert cert["hypothesis"] == "Personal information is transferred outside South Africa"

    def test_records_judge_id(self):
        engine = AuditEngine()
        cert = engine.record(
            intent="x", output="o", score=0.5, passed=True, judge_id="POPIAJudge/v1@0.75"
        )
        assert cert["judge_id"] == "POPIAJudge/v1@0.75"

    def test_records_subject(self):
        engine = AuditEngine()
        cert = engine.record(
            intent="x", output="o", score=0.5, passed=True,
            subject="user:5b4c9d12",
        )
        assert cert["subject"] == "user:5b4c9d12"

    def test_records_metadata_dict(self):
        engine = AuditEngine()
        meta = {"destination": "Ashby", "country": "US", "threshold": 0.75}
        cert = engine.record(intent="x", output="o", score=0.5, passed=True, metadata=meta)
        assert cert["metadata"] == meta

    def test_new_certs_are_v2(self):
        engine = AuditEngine()
        cert = engine.record(intent="x", output="o", score=0.5, passed=True)
        assert cert["@context"] == "https://schema.semantix.ai/v2"

    def test_old_style_call_still_works_new_fields_none(self):
        # Backward compat: 0.2.3 signature keeps working; new fields default to None.
        engine = AuditEngine()
        cert = engine.record(intent="x", output="o", score=0.5, passed=True, reason="r")
        assert cert["hypothesis"] is None
        assert cert["judge_id"] is None
        assert cert["subject"] is None
        assert cert["metadata"] is None

    def test_new_fields_are_keyword_only(self):
        import inspect

        params = inspect.signature(AuditEngine.record).parameters
        for name in ("hypothesis", "judge_id", "subject", "metadata"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is None


# ---------------------------------------------------------------------------
# F4 — claim_hash ADDED alongside output_hash (never redefine output_hash)
#
# output_hash = sha256(premise) is KEPT — two certs judging different clauses
# against the same premise still collapse to one output_hash, which made "the
# consent basis is identical for every user" cryptographically visible in the
# production audit. claim_hash = sha256(premise, hypothesis, judge_id) is ADDED
# so you can also tell WHAT was judged.
# ---------------------------------------------------------------------------


class TestClaimHash:
    def test_output_hash_is_still_premise_only(self):
        engine = AuditEngine()
        cert = engine.record(
            intent="x", output="hello world", score=0.5, passed=True,
            hypothesis="something entirely different",
        )
        assert cert["output_hash"] == hashlib.sha256(b"hello world").hexdigest()

    def test_claim_hash_present_when_hypothesis_given(self):
        engine = AuditEngine()
        cert = engine.record(
            intent="x", output="o", score=0.5, passed=True, hypothesis="h", judge_id="j"
        )
        assert cert["claim_hash"] is not None
        assert len(cert["claim_hash"]) == 64  # sha256 hex

    def test_claim_hash_none_without_hypothesis(self):
        engine = AuditEngine()
        cert = engine.record(intent="x", output="o", score=0.5, passed=True)
        assert cert["claim_hash"] is None

    def test_same_premise_different_hypothesis_distinct_claim_hash(self):
        engine = AuditEngine()
        a = engine.record(intent="x", output="same premise", score=0.5, passed=True,
                          hypothesis="clause A", judge_id="j")
        b = engine.record(intent="x", output="same premise", score=0.5, passed=True,
                          hypothesis="clause B", judge_id="j")
        assert a["claim_hash"] != b["claim_hash"]

    def test_same_premise_different_hypothesis_same_output_hash(self):
        # The constancy diagnostic (F4 tension) must be preserved.
        engine = AuditEngine()
        a = engine.record(intent="x", output="same premise", score=0.5, passed=True,
                          hypothesis="clause A", judge_id="j")
        b = engine.record(intent="x", output="same premise", score=0.5, passed=True,
                          hypothesis="clause B", judge_id="j")
        assert a["output_hash"] == b["output_hash"]

    def test_claim_hash_depends_on_judge_id(self):
        engine = AuditEngine()
        a = engine.record(intent="x", output="p", score=0.5, passed=True,
                          hypothesis="h", judge_id="v1")
        b = engine.record(intent="x", output="p", score=0.5, passed=True,
                          hypothesis="h", judge_id="v2")
        assert a["claim_hash"] != b["claim_hash"]


# ---------------------------------------------------------------------------
# Hard constraint — a v1 chain written yesterday verifies tomorrow
# ---------------------------------------------------------------------------


class TestBackwardCompatV1Chain:
    def _load_fixture(self):
        entries = []
        with open(_FIXTURES / "v1_chain_frozen.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def test_frozen_v1_fixture_still_verifies(self):
        entries = self._load_fixture()
        assert len(entries) == 5
        assert entries[0]["@context"] == "https://schema.semantix.ai/v1"
        assert AuditEngine.verify_entries(entries) is True

    def test_frozen_v1_fixture_tamper_detected(self):
        entries = self._load_fixture()
        entries[2]["score"] = 9.99
        assert AuditEngine.verify_entries(entries) is False

    def test_load_replaces_entries_and_verifies(self):
        engine = AuditEngine()
        engine.load(_FIXTURES / "v1_chain_frozen.jsonl")
        assert len(engine.entries) == 5
        assert engine.verify_chain() is True

    def test_mixed_v1_v2_chain_verifies(self):
        # Real deployments upgrade mid-life: v1 rows, then v2 rows appended.
        engine = AuditEngine()
        engine.load(_FIXTURES / "v1_chain_frozen.jsonl")
        frozen_first = dict(engine.entries[0])
        engine.record(intent="POPIA §72", output="new premise", score=0.5, passed=False,
                     hypothesis="Data leaves SA", judge_id="POPIAJudge/v1")
        engine.record(intent="POPIA §72", output="another", score=0.6, passed=True,
                     hypothesis="Consent given", judge_id="POPIAJudge/v1")
        assert engine.verify_chain() is True
        assert engine.entries[5]["@context"] == "https://schema.semantix.ai/v2"
        # appending v2 must not have mutated the frozen v1 entries
        assert engine.entries[0] == frozen_first

    def test_first_v2_entry_links_to_last_v1_entry(self):
        engine = AuditEngine()
        engine.load(_FIXTURES / "v1_chain_frozen.jsonl")
        last_v1 = engine.entries[-1]
        engine.record(intent="x", output="o", score=0.5, passed=True, hypothesis="h")
        expected = hashlib.sha256(json.dumps(last_v1, sort_keys=True).encode()).hexdigest()
        assert engine.entries[5]["previous_hash"] == expected


# ---------------------------------------------------------------------------
# F-NEW — distinct-verdict / distinct-claim surfacing
#
# The audit emitted 3,332 certs with verify_chain()==True and only 4 distinct
# verdicts. "The validation passes while the validated thing is constant." A
# chain that verifies perfectly while certifying one repeated fact should SAY so.
# ---------------------------------------------------------------------------


class TestChainReport:
    def test_counts_distinct_verdicts(self):
        engine = AuditEngine()
        for s in (0.1, 0.1, 0.9, 0.1):
            engine.record(intent="x", output=f"o{s}", score=s, passed=s > 0.5)
        r = engine.chain_report()
        assert r.n_certs == 4
        assert r.distinct_verdicts == 2  # (False,0.1) and (True,0.9)

    def test_constant_chain_is_flagged(self):
        engine = AuditEngine()
        for i in range(100):
            engine.record(intent="x", output=f"o{i}", score=0.5, passed=True)
        r = engine.chain_report()
        assert r.n_certs == 100
        assert r.distinct_verdicts == 1
        assert r.is_constant is True

    def test_varied_chain_not_flagged(self):
        engine = AuditEngine()
        for i in range(10):
            engine.record(intent="x", output=f"o{i}", score=i / 10, passed=i > 5)
        r = engine.chain_report()
        assert r.is_constant is False

    def test_single_cert_not_flagged_constant(self):
        engine = AuditEngine()
        engine.record(intent="x", output="o", score=0.5, passed=True)
        assert engine.chain_report().is_constant is False

    def test_report_valid_reflects_verification(self):
        engine = AuditEngine()
        engine.record(intent="x", output="a", score=0.5, passed=True)
        engine.record(intent="y", output="b", score=0.6, passed=True)
        assert engine.chain_report().valid is True
        engine.entries[0]["score"] = 9.99
        assert engine.chain_report().valid is False

    def test_distinct_premises_counts_output_hash(self):
        engine = AuditEngine()
        # same premise judged against two clauses -> 1 premise, 2 claims
        engine.record(intent="x", output="same", score=0.5, passed=True,
                     hypothesis="A", judge_id="j")
        engine.record(intent="x", output="same", score=0.5, passed=True,
                     hypothesis="B", judge_id="j")
        r = engine.chain_report()
        assert r.distinct_premises == 1
        assert r.distinct_claims == 2

    def test_v1_claims_fall_back_to_output_hash(self):
        # v1 certs have no claim_hash; claim identity falls back to output_hash.
        engine = AuditEngine()
        engine.load(_FIXTURES / "v1_chain_frozen.jsonl")
        r = engine.chain_report()
        assert r.distinct_claims == 5  # 5 distinct premises in the fixture

    def test_constant_report_str_warns(self):
        engine = AuditEngine()
        for i in range(50):
            engine.record(intent="x", output=f"o{i}", score=0.5, passed=True)
        text = str(engine.chain_report()).lower()
        assert "constant" in text

    def test_report_str_is_ascii_safe(self):
        # The report prints to consoles that may be cp1252 (Windows). A non-ASCII
        # glyph in the warning crashes there even though pytest's UTF-8 capture
        # hides it. Keep the human-facing string ASCII-encodable.
        engine = AuditEngine()
        for i in range(3):
            engine.record(intent="x", output=f"o{i}", score=0.5, passed=True)
        str(engine.chain_report()).encode("ascii")  # must not raise


class TestVerifyEntriesStatic:
    def test_empty_list_is_valid(self):
        assert AuditEngine.verify_entries([]) is True

    def test_static_verify_matches_instance(self):
        engine = AuditEngine()
        for i in range(3):
            engine.record(intent=f"i{i}", output=f"o{i}", score=0.5, passed=True)
        assert AuditEngine.verify_entries(engine.entries) == engine.verify_chain()
