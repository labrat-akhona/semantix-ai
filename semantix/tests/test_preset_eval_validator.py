"""Tests for scripts/validate_popia_preset_eval.py, the preset label set's gatekeeper."""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

# scripts/ is not a package — load the validator by path.
_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate_popia_preset_eval.py"
_spec = importlib.util.spec_from_file_location("validate_popia_preset_eval", _SCRIPT)
vpe = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves its module through sys.modules, and on Python 3.14
# an unregistered module makes dataclasses raise AttributeError on None.
sys.modules[_spec.name] = vpe
_spec.loader.exec_module(vpe)


def _authored_rows():
    """A complete, clean set: every slot filled with a unique, detailed premise."""
    rows = vpe.scaffold_rows()
    for i, row in enumerate(rows):
        row["scenario"] = f"scenario {i}"
        row["premise"] = (
            f"Case {i}: the internal memo records how the organisation handled "
            f"customer records in situation number {i} during the review period."
        )
    return rows


def test_preset_map_matches_the_presets_module():
    from semantix.presets import popia

    assert set(vpe.PRESET_CLAUSES) == set(popia.__all__)
    for name, clause in vpe.PRESET_CLAUSES.items():
        assert getattr(popia, name).clause == clause


def test_scaffold_has_seventy_balanced_slots():
    rows = vpe.scaffold_rows()
    assert len(rows) == 70
    for preset in vpe.PRESET_CLAUSES:
        mine = [r for r in rows if r["preset"] == preset]
        labels = collections.Counter(r["expected"] for r in mine)
        assert labels == {"complies": 4, "violates": 4, "insufficient": 2}
        styles = collections.Counter(r["style"] for r in mine)
        assert all(styles[s] >= 2 for s in vpe.STYLES)
    assert all(r["premise"] == "TODO" and r["scenario"] == "TODO" for r in rows)


def test_unwritten_slots_are_reported_as_todo():
    report = vpe.validate(vpe.scaffold_rows(), leak_premises=set())
    assert len(report.todo) == 70
    assert report.clean is False


def test_clean_set_has_no_errors_or_warnings():
    report = vpe.validate(_authored_rows(), leak_premises=set())
    assert report.clean
    assert report.warnings == []


def test_leaked_premise_is_an_error():
    rows = _authored_rows()
    rows[3]["premise"] = "An existing training premise about consent, restated as a memo."
    report = vpe.validate(rows, {vpe.norm(rows[3]["premise"])})
    assert any("row 3" in e and "leak" in e for e in report.errors)


def test_duplicate_premise_is_an_error():
    rows = _authored_rows()
    rows[5]["premise"] = rows[4]["premise"]
    report = vpe.validate(rows, set())
    assert any("row 5" in e and "duplicates row 4" in e for e in report.errors)


def test_changed_label_breaks_the_split():
    rows = _authored_rows()
    rows[0]["expected"] = "violates"
    report = vpe.validate(rows, set())
    assert any(e.startswith("POPIA_CONSENT: label split") for e in report.errors)


def test_clause_must_match_its_preset():
    rows = _authored_rows()
    rows[0]["clause"] = "POPIA security safeguards"
    report = vpe.validate(rows, set())
    assert any("row 0" in e and "does not match POPIA_CONSENT" in e for e in report.errors)


def test_short_premise_warns_but_does_not_block():
    rows = _authored_rows()
    rows[2]["premise"] = "I agree to the privacy terms."
    report = vpe.validate(rows, set())
    assert report.clean
    assert any("row 2" in w for w in report.warnings)


def test_leak_sources_are_read_from_existing_data(tmp_path, monkeypatch):
    seeds = tmp_path / "popia_seeds.jsonl"
    seeds.write_text('{"premise": "A Seed Premise."}\n{"no_premise": 1}\n', encoding="utf-8")
    monkeypatch.setattr(vpe, "LEAK_SOURCES", (str(seeds),))
    monkeypatch.setattr(vpe, "EVAL_PATH", tmp_path / "popia_preset_eval.jsonl")
    assert vpe.load_leak_premises() == {"a seed premise"}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    eval_path = tmp_path / "popia_preset_eval.jsonl"
    hash_path = tmp_path / "hash.txt"
    seeds = tmp_path / "popia_seeds.jsonl"
    seeds.write_text(json.dumps({"premise": "An existing premise."}) + "\n", encoding="utf-8")
    monkeypatch.setattr(vpe, "EVAL_PATH", eval_path)
    monkeypatch.setattr(vpe, "HASH_PATH", hash_path)
    monkeypatch.setattr(vpe, "LEAK_SOURCES", (str(seeds),))
    return eval_path, hash_path


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_scaffold_writes_once_and_never_overwrites(paths):
    eval_path, _hash_path = paths
    assert vpe.main(["--scaffold"]) == 0
    assert len(eval_path.read_text(encoding="utf-8").splitlines()) == 70
    eval_path.write_text("authored work\n", encoding="utf-8")
    assert vpe.main(["--scaffold"]) == 1
    assert eval_path.read_text(encoding="utf-8") == "authored work\n"


def test_check_without_a_file_asks_for_the_scaffold(paths, capsys):
    assert vpe.main([]) == 1
    assert "--scaffold" in capsys.readouterr().out


def test_unfinished_file_is_never_pinned(paths):
    _eval_path, hash_path = paths
    vpe.main(["--scaffold"])
    assert vpe.main(["--pin"]) == 1
    assert not hash_path.exists()


def test_complete_clean_file_is_pinned(paths):
    eval_path, hash_path = paths
    _write(eval_path, _authored_rows())
    assert vpe.main(["--pin"]) == 0
    assert hash_path.read_text().strip() == hashlib.sha256(eval_path.read_bytes()).hexdigest()
