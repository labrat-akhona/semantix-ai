"""SQLite-backed JudgeResult cache keyed on SHA-256(judge, text, intent)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from benchmarks.common.judges import JudgeResult


class JudgeCache:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, "
            "score REAL NOT NULL, "
            "latency_ms REAL NOT NULL, "
            "cost_usd REAL NOT NULL, "
            "paid_equivalent_usd REAL NOT NULL, "
            "raw TEXT)"
        )
        self._conn.commit()

    def _key(self, judge: str, text: str, intent: str) -> str:
        blob = json.dumps([judge, text, intent], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, judge: str, text: str, intent: str) -> JudgeResult | None:
        row = self._conn.execute(
            "SELECT score, latency_ms, cost_usd, paid_equivalent_usd, raw FROM cache WHERE key=?",
            (self._key(judge, text, intent),),
        ).fetchone()
        if row is None:
            return None
        score, latency_ms, cost_usd, paid, raw = row
        return JudgeResult(
            score=score, latency_ms=latency_ms, cost_usd=cost_usd,
            paid_equivalent_usd=paid, raw=raw,
        )

    def put(self, judge: str, text: str, intent: str, result: JudgeResult) -> None:
        if result.error is not None:
            return  # Don't cache errors — retry may succeed
        self._conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._key(judge, text, intent),
                result.score,
                result.latency_ms,
                result.cost_usd,
                result.paid_equivalent_usd,
                result.raw,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
