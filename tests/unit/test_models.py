"""Unit tests for Pydantic data models — written before implementation."""

import pytest
from financial_qa.models.conversation import ConversationTurn, Question
from financial_qa.models.pipeline import (
    ExtractedValue,
    PipelineResult,
    PipelineStep,
    StepStatus,
)


class TestQuestion:
    def test_question_stores_text_and_turn(self) -> None:
        q = Question(text="What is the revenue?", turn=1, conversation_id="conv-1")
        assert q.text == "What is the revenue?"
        assert q.turn == 1
        assert q.conversation_id == "conv-1"

    def test_question_turn_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Question(text="x", turn=0, conversation_id="c")

    def test_question_text_cannot_be_empty(self) -> None:
        with pytest.raises(ValueError):
            Question(text="", turn=1, conversation_id="c")


class TestConversationTurn:
    def test_turn_stores_question_and_answer(self) -> None:
        turn = ConversationTurn(
            question="What is the revenue?",
            answer="14.5 million",
            numerical_answer=14.5,
        )
        assert turn.question == "What is the revenue?"
        assert turn.numerical_answer == 14.5

    def test_turn_answer_can_be_none(self) -> None:
        turn = ConversationTurn(question="What is revenue?", answer=None)
        assert turn.answer is None
        assert turn.numerical_answer is None


class TestExtractedValue:
    def test_extracted_value_stores_name_and_value(self) -> None:
        ev = ExtractedValue(name="revenue", raw="$14.5M", normalised=14.5)
        assert ev.name == "revenue"
        assert ev.normalised == 14.5

    def test_normalised_value_is_float(self) -> None:
        ev = ExtractedValue(name="x", raw="10%", normalised=0.10)
        assert isinstance(ev.normalised, float)


class TestPipelineStep:
    def test_step_captures_name_input_output(self) -> None:
        step = PipelineStep(
            name="rewriter",
            status=StepStatus.SUCCESS,
            input="raw question",
            output="rewritten question",
        )
        assert step.name == "rewriter"
        assert step.status == StepStatus.SUCCESS

    def test_failed_step_stores_error(self) -> None:
        step = PipelineStep(
            name="executor",
            status=StepStatus.FAILED,
            input="1 / 0",
            output=None,
            error="ZeroDivisionError",
        )
        assert step.status == StepStatus.FAILED
        assert step.error == "ZeroDivisionError"


class TestPipelineResult:
    def test_result_with_answer(self) -> None:
        result = PipelineResult(
            conversation_id="c1",
            turn=1,
            question="What is margin?",
            answer=0.32,
            expression="14.5 / 45.3",
            steps=[],
        )
        assert result.answer == pytest.approx(0.32)
        assert result.expression == "14.5 / 45.3"

    def test_result_can_be_failed(self) -> None:
        result = PipelineResult(
            conversation_id="c1",
            turn=1,
            question="What is margin?",
            answer=None,
            expression=None,
            steps=[],
            error="extraction_failed",
        )
        assert result.answer is None
        assert result.error == "extraction_failed"
