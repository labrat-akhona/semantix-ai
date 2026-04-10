"""Tests for the Semantix MCP server.

Verifies tool registration, response schema, and graceful error handling
without requiring the NLI model to be loaded.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from semantix.judges import Verdict
from semantix.mcp.server import (
    _build_correction,
    mcp,
    verify_text_intent,
)

# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_server_name(self):
        assert mcp.name == "Semantix-Verify"

    def test_verify_text_intent_registered(self):
        tools = list(mcp._tool_manager._tools.keys())
        assert "verify_text_intent" in tools

    def test_only_one_tool_registered(self):
        tools = list(mcp._tool_manager._tools.keys())
        assert len(tools) == 1


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def _mock_judge(passed: bool, score: float, reason: str | None = None):
    """Return a mock NLIJudge that returns a fixed Verdict."""
    judge = MagicMock()
    judge.evaluate.return_value = Verdict(passed=passed, score=score, reason=reason)
    return judge


class TestResponseSchema:
    @patch("semantix.mcp.server._get_judge")
    def test_passing_response_has_required_keys(self, mock_get):
        mock_get.return_value = _mock_judge(passed=True, score=0.92)
        result = json.loads(verify_text_intent("good text", "some intent"))
        assert "score" in result
        assert "passed" in result
        assert "reason" in result

    @patch("semantix.mcp.server._get_judge")
    def test_passing_response_has_no_correction(self, mock_get):
        mock_get.return_value = _mock_judge(passed=True, score=0.92)
        result = json.loads(verify_text_intent("good text", "some intent"))
        assert "correction_suggestion" not in result

    @patch("semantix.mcp.server._get_judge")
    def test_failing_response_includes_correction(self, mock_get):
        mock_get.return_value = _mock_judge(passed=False, score=0.15)
        result = json.loads(verify_text_intent("bad text", "some intent"))
        assert result["passed"] is False
        assert "correction_suggestion" in result

    @patch("semantix.mcp.server._get_judge")
    def test_score_is_float(self, mock_get):
        mock_get.return_value = _mock_judge(passed=True, score=0.88)
        result = json.loads(verify_text_intent("text", "intent"))
        assert isinstance(result["score"], float)

    @patch("semantix.mcp.server._get_judge")
    def test_passed_is_bool(self, mock_get):
        mock_get.return_value = _mock_judge(passed=True, score=0.88)
        result = json.loads(verify_text_intent("text", "intent"))
        assert isinstance(result["passed"], bool)

    @patch("semantix.mcp.server._get_judge")
    def test_reason_forwarded_from_verdict(self, mock_get):
        mock_get.return_value = _mock_judge(passed=False, score=0.2, reason="entailment too low")
        result = json.loads(verify_text_intent("text", "intent"))
        assert result["reason"] == "entailment too low"

    @patch("semantix.mcp.server._get_judge")
    def test_custom_threshold_passed_to_judge(self, mock_get):
        judge = _mock_judge(passed=True, score=0.6)
        mock_get.return_value = judge
        verify_text_intent("text", "intent", threshold=0.3)
        judge.evaluate.assert_called_once_with("text", "intent", 0.3)

    @patch("semantix.mcp.server._get_judge")
    def test_return_value_is_valid_json_string(self, mock_get):
        mock_get.return_value = _mock_judge(passed=True, score=0.95)
        raw = verify_text_intent("text", "intent")
        assert isinstance(raw, str)
        json.loads(raw)  # must not raise


# ---------------------------------------------------------------------------
# Correction suggestion
# ---------------------------------------------------------------------------


class TestBuildCorrection:
    def test_contains_score(self):
        c = _build_correction("bad", "be good", 0.1234, 0.5)
        assert "0.1234" in c

    def test_contains_threshold(self):
        c = _build_correction("bad", "be good", 0.1, 0.8)
        assert "0.8" in c

    def test_contains_requirement(self):
        c = _build_correction("bad", "be good", 0.1, 0.5)
        assert "be good" in c

    def test_contains_rejected_output(self):
        c = _build_correction("my bad output", "be good", 0.1, 0.5)
        assert "my bad output" in c

    def test_truncates_long_output(self):
        long_text = "x" * 500
        c = _build_correction(long_text, "intent", 0.1, 0.5)
        # Output truncated to 400 chars in the code block
        assert "x" * 400 in c
        assert "x" * 401 not in c

    def test_contains_action_prompt(self):
        c = _build_correction("bad", "be good", 0.1, 0.5)
        assert "Please generate a new response" in c


# ---------------------------------------------------------------------------
# Dependency error handling
# ---------------------------------------------------------------------------


class TestDependencyHandling:
    @patch("semantix.mcp.server._get_judge", side_effect=ImportError("no module"))
    def test_import_error_returns_json(self, mock_get):
        raw = verify_text_intent("text", "intent")
        result = json.loads(raw)
        assert result["passed"] is False
        assert "error" in result

    @patch("semantix.mcp.server._get_judge", side_effect=ImportError("no module"))
    def test_import_error_suggests_install(self, mock_get):
        result = json.loads(verify_text_intent("text", "intent"))
        assert "pip install" in result["error"]

    @patch("semantix.mcp.server._get_judge", side_effect=ImportError("no module"))
    def test_import_error_does_not_crash(self, mock_get):
        """Server must stay alive even when dependencies are missing."""
        # Should not raise — returns error JSON instead
        raw = verify_text_intent("text", "intent")
        assert isinstance(raw, str)
