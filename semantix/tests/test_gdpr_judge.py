"""Unit tests for GDPRJudge (scaffold judge; fine-tuned weights unpublished)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from semantix.judges import Verdict


def _fake_session():
    session = MagicMock()
    ids, mask = MagicMock(), MagicMock()
    ids.name, mask.name = "input_ids", "attention_mask"
    session.get_inputs.return_value = [ids, mask]
    # Label order: contradiction, entailment, neutral. softmax(...)[1] is about 0.875.
    session.run.return_value = [np.array([[0.2, 3.0, 0.5]], dtype=np.float32)]
    return session


def _fake_tokenizer():
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids, encoded.attention_mask, encoded.type_ids = [1, 2, 3], [1, 1, 1], [0, 0, 0]
    tokenizer.encode.return_value = encoded
    return tokenizer


@pytest.fixture
def mocked_models(monkeypatch):
    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_session",
        lambda variant, repo_id=None: _fake_session(),
    )
    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_tokenizer",
        lambda repo_id=None: _fake_tokenizer(),
    )


def test_evaluate_returns_verdict(mocked_models):
    from semantix.judges.gdpr import GDPRJudge

    verdict = GDPRJudge().evaluate("premise text", "hypothesis text", threshold=0.5)
    assert isinstance(verdict, Verdict)
    assert verdict.score == pytest.approx(0.875, abs=1e-3)
    assert verdict.passed is True


def test_scores_are_uncalibrated(mocked_models):
    from semantix.judges.gdpr import GDPRJudge

    assert GDPRJudge().calibrated is False


def test_falls_back_to_popia_weights_when_gdpr_repo_is_missing(monkeypatch):
    def load_session(variant, repo_id=None):
        if repo_id == "labrat-aiko/nli-gdpr-v1":
            raise FileNotFoundError("repo not published")
        return _fake_session()

    monkeypatch.setattr("semantix.judges.quantized_nli._load_session", load_session)
    monkeypatch.setattr(
        "semantix.judges.quantized_nli._load_tokenizer",
        lambda repo_id=None: _fake_tokenizer(),
    )
    from semantix.judges.gdpr import GDPRJudge

    with pytest.warns(RuntimeWarning, match="falling back"):
        judge = GDPRJudge()
    assert judge._repo_id == "labrat-aiko/nli-popia-v1"
    assert judge.evaluate("premise", "hypothesis", threshold=0.5).passed is True
