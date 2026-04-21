"""Judge-comparison metrics: Pearson r (continuous), Cohen's kappa (binary)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient, NaN-tolerant.

    Drops any (x, y) pair where either is NaN. Returns NaN if < 2 valid pairs
    or if either series has zero variance.
    """
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return float("nan")
    xs_c, ys_c = zip(*pairs, strict=True)
    mx = sum(xs_c) / len(xs_c)
    my = sum(ys_c) / len(ys_c)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs_c))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys_c))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def cohen_kappa_binary(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters.

    Returns 0.0 when chance agreement equals observed (no information beyond
    the base rate).
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must have equal length")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if expected == 1.0:
        return 0.0  # no variance; kappa conventionally 0
    return (observed - expected) / (1 - expected)
