"""Failure taxonomy for classifying pipeline errors.

Classifying failures by type enables targeted iteration: extraction failures
call for better prompting or context chunking; execution errors call for
tighter expression constraints; reasoning failures indicate prompt changes.
"""

from enum import StrEnum

from financial_qa.models.pipeline import PipelineResult


class FailureReason(StrEnum):
    """Top-level failure categories."""

    EXTRACTION_FAILURE = "extraction_failure"
    REASONING_FAILURE = "reasoning_failure"
    EXECUTION_ERROR = "execution_error"
    WRONG_ANSWER = "wrong_answer"


_EXECUTION_KEYWORDS = ("division by zero", "forbidden", "unparseable", "execution")
_EXTRACTION_KEYWORDS = ("extraction", "no values", "json")


def classify_failure(result: PipelineResult) -> FailureReason | None:
    """Classify the failure reason for a pipeline result.

    Args:
        result: A completed (or failed) PipelineResult.

    Returns:
        A FailureReason enum value, or None if the result was successful.
    """
    if result.answer is not None:
        return None

    error = (result.error or "").lower()

    if not error:
        return FailureReason.REASONING_FAILURE

    if result.expression is None and any(k in error for k in _EXTRACTION_KEYWORDS):
        return FailureReason.EXTRACTION_FAILURE

    if result.expression is None:
        return FailureReason.REASONING_FAILURE

    if any(k in error for k in _EXECUTION_KEYWORDS):
        return FailureReason.EXECUTION_ERROR

    return FailureReason.WRONG_ANSWER
