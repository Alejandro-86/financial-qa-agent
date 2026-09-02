"""Unit tests for the evaluation framework — metrics and failure taxonomy."""

import pytest
from financial_qa.eval.metrics import (
    ExecutionAccuracy,
    MeanAbsolutePercentageError,
    is_correct,
)
from financial_qa.eval.taxonomy import FailureReason, classify_failure
from financial_qa.models.pipeline import PipelineResult


class TestIsCorrect:
    """Numerical tolerance logic for execution accuracy."""

    def test_exact_match(self) -> None:
        assert is_correct(predicted=14.5, gold=14.5) is True

    def test_within_tolerance(self) -> None:
        assert is_correct(predicted=14.50001, gold=14.5, tolerance=1e-4) is True

    def test_outside_tolerance(self) -> None:
        assert is_correct(predicted=14.6, gold=14.5, tolerance=1e-4) is False

    def test_percentage_normalisation(self) -> None:
        # 20% stored as 0.20 vs answer of 20.0
        assert is_correct(predicted=0.20, gold=20.0, check_percentage_scale=True) is True

    def test_negative_values(self) -> None:
        assert is_correct(predicted=-3.5, gold=-3.5) is True

    def test_zero_gold(self) -> None:
        assert is_correct(predicted=0.0, gold=0.0) is True


class TestExecutionAccuracy:
    def test_all_correct(self) -> None:
        pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        ea = ExecutionAccuracy.compute(pairs)
        assert ea.accuracy == pytest.approx(1.0)
        assert ea.correct == 3
        assert ea.total == 3

    def test_none_correct(self) -> None:
        pairs = [(1.0, 2.0), (3.0, 4.0)]
        ea = ExecutionAccuracy.compute(pairs)
        assert ea.accuracy == pytest.approx(0.0)

    def test_partial_correct(self) -> None:
        pairs = [(1.0, 1.0), (999.0, 2.0)]
        ea = ExecutionAccuracy.compute(pairs)
        assert ea.accuracy == pytest.approx(0.5)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ExecutionAccuracy.compute([])

    def test_by_turn_depth(self) -> None:
        # turn 1 correct, turn 2 wrong
        results = [
            (1.0, 1.0, 1),
            (99.0, 2.0, 2),
        ]
        by_depth = ExecutionAccuracy.by_turn(results)
        assert by_depth[1].accuracy == pytest.approx(1.0)
        assert by_depth[2].accuracy == pytest.approx(0.0)


class TestMAPE:
    def test_perfect_prediction(self) -> None:
        pairs = [(10.0, 10.0), (20.0, 20.0)]
        assert MeanAbsolutePercentageError.compute(pairs) == pytest.approx(0.0)

    def test_known_error(self) -> None:
        # |10-8|/8 = 25%, |20-25|/25 = 20% → mean = 22.5%
        pairs = [(10.0, 8.0), (20.0, 25.0)]
        mape = MeanAbsolutePercentageError.compute(pairs)
        assert mape == pytest.approx(22.5, rel=1e-3)

    def test_skips_zero_gold(self) -> None:
        # zero gold is undefined for MAPE — should skip gracefully
        pairs = [(1.0, 0.0), (10.0, 10.0)]
        mape = MeanAbsolutePercentageError.compute(pairs)
        assert mape == pytest.approx(0.0)


class TestFailureTaxonomy:
    def test_classifies_executor_error(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=None, expression="1/0", error="division by zero", steps=[]
        )
        assert classify_failure(result) == FailureReason.EXECUTION_ERROR

    def test_classifies_missing_expression(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=None, expression=None, error="reasoner returned empty", steps=[]
        )
        assert classify_failure(result) == FailureReason.REASONING_FAILURE

    def test_classifies_extraction_failure(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=None, expression=None, error="extraction_failed", steps=[]
        )
        assert classify_failure(result) == FailureReason.EXTRACTION_FAILURE

    def test_no_failure_for_successful_result(self) -> None:
        result = PipelineResult(
            conversation_id="c1", turn=1, question="q",
            answer=14.5, expression="14.5", steps=[]
        )
        assert classify_failure(result) is None
