"""Unit tests for semantix.presets.popia."""

from __future__ import annotations

from semantix.intent import Intent
from semantix.judges.popia import POPIAJudge


def _all_presets():
    from semantix.presets import popia as m
    return [getattr(m, n) for n in m.__all__]


def test_all_presets_are_intents():
    for preset in _all_presets():
        assert isinstance(preset, Intent)


def test_all_presets_have_nonempty_description():
    for preset in _all_presets():
        assert preset.description
        assert len(preset.description) > 10


def test_all_presets_have_clause_attribute_matching_judge():
    canonical = set(POPIAJudge.clauses())
    for preset in _all_presets():
        assert hasattr(preset, "clause")
        assert preset.clause in canonical, f"{preset.clause!r} not in canonical"


def test_breach_preset_is_negated():
    from semantix.presets.popia import POPIA_BREACH
    assert POPIA_BREACH.negate is True


def test_non_breach_presets_are_not_negated():
    from semantix.presets import popia as m
    for name in m.__all__:
        if name == "POPIA_BREACH":
            continue
        preset = getattr(m, name)
        assert preset.negate is False, f"{name} should not be negated"


def test_security_preset_has_stricter_threshold():
    from semantix.presets.popia import POPIA_SECURITY
    assert POPIA_SECURITY.threshold is not None
    assert 0.8 < POPIA_SECURITY.threshold <= 0.95


def test_all_thresholds_in_valid_range():
    for preset in _all_presets():
        if preset.threshold is not None:
            assert 0.5 <= preset.threshold <= 0.95


def test_preset_count_matches_clause_count():
    from semantix.presets import popia as m
    assert len(m.__all__) == len(POPIAJudge.clauses())


def test_preset_clauses_cover_all_judge_clauses():
    from semantix.presets import popia as m
    covered = {getattr(m, n).clause for n in m.__all__}
    assert covered == set(POPIAJudge.clauses())
