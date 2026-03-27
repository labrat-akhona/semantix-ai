"""Caching wrapper for any Judge implementation."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from semantix.judges import Judge, Verdict


class CachingJudge(Judge):
    """LRU-cache wrapper around any ``Judge`` backend.

    Caches verdicts keyed on ``(output_hash, intent_description_hash,
    threshold)`` so identical evaluations are never run twice.

    Parameters
    ----------
    judge:
        The underlying judge to delegate to on cache misses.
    maxsize:
        Maximum number of cached verdicts (LRU eviction).
    """

    def __init__(self, judge: Judge, maxsize: int = 256) -> None:
        self._judge = judge
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Verdict] = OrderedDict()

    # ── stats ───────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    _hits: int = 0
    _misses: int = 0

    # ── Judge interface ─────────────────────────────────────────

    def evaluate(
        self,
        output: str,
        intent_description: str,
        threshold: float = 0.8,
    ) -> Verdict:
        key = self._make_key(output, intent_description, threshold)
        if key in self._cache:
            self._hits += 1
            # Move to end (most-recently-used).
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        verdict = self._judge.evaluate(output, intent_description, threshold)
        self._cache[key] = verdict
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict LRU
        return verdict

    def clear(self) -> None:
        """Flush the entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # ── internals ───────────────────────────────────────────────

    @staticmethod
    def _make_key(output: str, description: str, threshold: float) -> str:
        h = hashlib.sha256()
        h.update(output.encode())
        h.update(description.encode())
        h.update(str(threshold).encode())
        return h.hexdigest()
