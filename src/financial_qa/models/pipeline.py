"""Pipeline-level data models for tracking step inputs, outputs, and results."""

from enum import StrEnum

from pydantic import BaseModel


class StepStatus(StrEnum):
    """Execution status of a single pipeline step."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExtractedValue(BaseModel):
    """A single numerical value extracted from the financial context.

    Args:
        name: Human-readable label for this value (e.g. 'revenue_2023').
        raw: The raw string as it appeared in the source (e.g. '$14.5M').
        normalised: The value normalised to a plain float.
    """

    name: str
    raw: str
    normalised: float


class PipelineStep(BaseModel):
    """Record of a single step's execution within the pipeline.

    Args:
        name: Step name — one of rewriter, extractor, reasoner, executor.
        status: Whether the step succeeded, failed, or was skipped.
        input: The string passed into this step.
        output: The string produced by this step, or None on failure.
        error: Error message if status is FAILED, else None.
        latency_ms: Wall-clock time for this step in milliseconds.
    """

    name: str
    status: StepStatus
    input: str
    output: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


class PipelineResult(BaseModel):
    """The complete result of running the four-step pipeline on one question.

    Args:
        conversation_id: Identifies the multi-turn conversation.
        turn: 1-based turn index.
        question: The original question text.
        answer: The final numerical answer as a float, or None on failure.
        expression: The Python arithmetic expression evaluated by the executor.
        steps: Ordered list of step records for observability.
        error: Top-level failure reason if the pipeline did not produce an answer.
    """

    conversation_id: str
    turn: int
    question: str
    answer: float | None = None
    expression: str | None = None
    steps: list[PipelineStep] = []
    error: str | None = None
