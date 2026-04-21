from pathlib import Path

from benchmarks.common.cache import JudgeCache
from benchmarks.common.judges import JudgeResult


def test_cache_miss_then_hit(tmp_path: Path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    assert cache.get("groq", "text", "intent") is None
    cache.put(
        "groq", "text", "intent",
        JudgeResult(score=0.9, latency_ms=100, cost_usd=0, paid_equivalent_usd=0.0001),
    )
    hit = cache.get("groq", "text", "intent")
    assert hit is not None
    assert hit.score == 0.9


def test_cache_key_discrimination(tmp_path: Path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    cache.put(
        "groq", "A", "intent",
        JudgeResult(score=0.1, latency_ms=0, cost_usd=0, paid_equivalent_usd=0),
    )
    cache.put(
        "groq", "B", "intent",
        JudgeResult(score=0.9, latency_ms=0, cost_usd=0, paid_equivalent_usd=0),
    )
    assert cache.get("groq", "A", "intent").score == 0.1
    assert cache.get("groq", "B", "intent").score == 0.9
    assert cache.get("semantix", "A", "intent") is None  # Different judge name


def test_cache_does_not_store_errored_results(tmp_path: Path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    cache.put(
        "groq", "text", "intent",
        JudgeResult(
            score=float("nan"), latency_ms=0, cost_usd=0,
            paid_equivalent_usd=0, error="non-numeric",
        ),
    )
    # Errors should NOT be cached — retry may succeed
    assert cache.get("groq", "text", "intent") is None
