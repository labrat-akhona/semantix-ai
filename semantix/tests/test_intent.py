"""Tests for the Intent base class."""

import pytest

from semantix.intent import Intent


class Polite(Intent):
    """The text must be polite and friendly."""


class NoDoc(Intent):
    pass


class EmptyDoc(Intent):
    """ """


def test_description_returns_docstring():
    assert Polite.description() == "The text must be polite and friendly."


def test_missing_docstring_raises():
    with pytest.raises(TypeError, match="non-empty docstring"):
        NoDoc.description()


def test_empty_docstring_raises():
    with pytest.raises(TypeError, match="non-empty docstring"):
        EmptyDoc.description()


def test_intent_wraps_text():
    p = Polite("hello")
    assert p.text == "hello"
    assert str(p) == "hello"
    assert repr(p) == "Polite('hello')"


def test_intent_equality():
    a = Polite("hello")
    b = Polite("hello")
    c = Polite("world")
    assert a == b
    assert a != c
    assert a == "hello"
    assert hash(a) == hash(b)


def test_default_threshold():
    assert Intent.threshold == 0.8


def test_custom_threshold():
    class Strict(Intent):
        """Must be very strict."""

        threshold = 0.95

    assert Strict.threshold == 0.95
