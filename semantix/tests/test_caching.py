"""Tests for CachingJudge."""

from semantix.judges import Verdict
from semantix.judges.caching import CachingJudge
from semantix.tests.conftest import MockJudge


def test_cache_hit():
    inner = MockJudge(passed=True, score=0.95)
    caching = CachingJudge(inner, maxsize=10)

    v1 = caching.evaluate("hello", "be polite", 0.8)
    v2 = caching.evaluate("hello", "be polite", 0.8)

    assert v1 == v2
    assert inner.call_count == 1  # only called once
    assert caching.hits == 1
    assert caching.misses == 1


def test_cache_miss_on_different_input():
    inner = MockJudge(passed=True, score=0.9)
    caching = CachingJudge(inner, maxsize=10)

    caching.evaluate("hello", "be polite", 0.8)
    caching.evaluate("goodbye", "be polite", 0.8)

    assert inner.call_count == 2
    assert caching.misses == 2


def test_lru_eviction():
    inner = MockJudge(passed=True, score=0.9)
    caching = CachingJudge(inner, maxsize=2)

    caching.evaluate("a", "desc", 0.8)
    caching.evaluate("b", "desc", 0.8)
    caching.evaluate("c", "desc", 0.8)  # evicts "a"

    # "a" should be evicted
    caching.evaluate("a", "desc", 0.8)  # miss
    assert inner.call_count == 4
    assert caching.misses == 4


def test_clear_resets():
    inner = MockJudge(passed=True)
    caching = CachingJudge(inner, maxsize=10)

    caching.evaluate("x", "y", 0.8)
    caching.clear()

    caching.evaluate("x", "y", 0.8)
    assert inner.call_count == 2
    assert caching.hits == 0
    assert caching.misses == 1
