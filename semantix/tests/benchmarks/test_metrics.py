import math

import pytest

from benchmarks.common.metrics import cohen_kappa_binary, pearson_r


def test_pearson_r_perfect_correlation():
    assert pearson_r([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_pearson_r_perfect_anti_correlation():
    assert pearson_r([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_r_skips_nan_pairs():
    r = pearson_r([1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0])
    assert r == pytest.approx(1.0)


def test_pearson_r_returns_nan_when_all_nan():
    assert math.isnan(pearson_r([float("nan")], [1.0]))


def test_cohen_kappa_perfect_agreement():
    a = [True, True, False, False]
    b = [True, True, False, False]
    assert cohen_kappa_binary(a, b) == pytest.approx(1.0)


def test_cohen_kappa_perfect_disagreement():
    a = [True, True, False, False]
    b = [False, False, True, True]
    assert cohen_kappa_binary(a, b) == pytest.approx(-1.0)


def test_cohen_kappa_handles_zero_variance():
    # Both raters agree trivially (all True) — chance agreement = observed, kappa undefined -> 0.0 by convention
    assert cohen_kappa_binary([True, True], [True, True]) == pytest.approx(0.0)
