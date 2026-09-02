"""Unit tests for API request/response schemas."""

import pytest

from api.schemas import AskRequest, AskResponse
from financial_qa.models.pipeline import PipelineResult


class TestAskRequest:
    def test_valid_request(self) -> None:
        req = AskRequest(question="What is the margin?", context="Revenue: $14.5M")
        assert req.question == "What is the margin?"
        assert req.conversation_id == "default"
        assert req.history == []

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError):
            AskRequest(question="", context="Revenue: $14.5M")

    def test_empty_context_raises(self) -> None:
        with pytest.raises(ValueError):
            AskRequest(question="What is margin?", context="")


class TestAskResponse:
    def test_from_successful_result(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=14.5, expression="14.5", steps=[],
        )
        resp = AskResponse.from_result(result)
        assert resp.answer == pytest.approx(14.5)
        assert resp.expression == "14.5"
        assert resp.cached is False

    def test_from_failed_result(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=None, expression=None, steps=[], error="extraction_failed",
        )
        resp = AskResponse.from_result(result, cached=True)
        assert resp.answer is None
        assert resp.error == "extraction_failed"
        assert resp.cached is True
