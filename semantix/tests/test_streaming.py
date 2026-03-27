"""Tests for StreamCollector."""

import asyncio

import pytest

from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.streaming import StreamCollector
from semantix.tests.conftest import MockJudge


class Polite(Intent):
    """The text must be polite."""


# ── sync context manager ───────────────────────────────────────────────


def test_stream_collector_context_manager_pass():
    judge = MockJudge(passed=True, score=0.9)
    with StreamCollector(Polite, judge=judge) as sc:
        sc.feed("Hello, ")
        sc.feed("how are you?")
    result = sc.result()
    assert isinstance(result, Polite)
    assert result.text == "Hello, how are you?"


def test_stream_collector_context_manager_fail():
    judge = MockJudge(passed=False, score=0.3)
    with StreamCollector(Polite, judge=judge) as sc:
        sc.feed("Get lost!")
    with pytest.raises(SemanticIntentError):
        sc.result()


# ── sync iterator wrapper ──────────────────────────────────────────────


def test_stream_wrap_sync():
    judge = MockJudge(passed=True, score=0.88)

    def fake_stream():
        yield "Thank "
        yield "you "
        yield "kindly!"

    sc = StreamCollector(Polite, judge=judge)
    chunks = list(sc.wrap(fake_stream()))
    assert chunks == ["Thank ", "you ", "kindly!"]
    result = sc.result()
    assert result.text == "Thank you kindly!"


# ── async context manager ──────────────────────────────────────────────


def test_async_stream_collector():
    judge = MockJudge(passed=True, score=0.9)

    async def run():
        async with StreamCollector(Polite, judge=judge) as sc:
            sc.feed("Nice ")
            sc.feed("day!")
        return sc.result()

    result = asyncio.run(run())
    assert isinstance(result, Polite)
    assert result.text == "Nice day!"


# ── async iterator wrapper ─────────────────────────────────────────────


def test_async_stream_wrap():
    judge = MockJudge(passed=True, score=0.9)

    async def fake_stream():
        for chunk in ["Good ", "morning!"]:
            yield chunk

    async def run():
        sc = StreamCollector(Polite, judge=judge)
        chunks = []
        async for chunk in sc.awrap(fake_stream()):
            chunks.append(chunk)
        return chunks, sc.result()

    chunks, result = asyncio.run(run())
    assert chunks == ["Good ", "morning!"]
    assert result.text == "Good morning!"


# ── text property ───────────────────────────────────────────────────────


def test_text_property_accumulates():
    judge = MockJudge(passed=True)
    sc = StreamCollector(Polite, judge=judge)
    sc.feed("a")
    sc.feed("b")
    assert sc.text == "ab"
