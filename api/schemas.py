"""Request and response schemas for the FastAPI application."""

from pydantic import BaseModel, field_validator

from financial_qa.models.conversation import ConversationTurn
from financial_qa.models.pipeline import PipelineResult


class AskRequest(BaseModel):
    """Request body for POST /ask.

    Args:
        question: The financial question to answer.
        context: The financial table or passage to reason over.
        conversation_id: Groups turns in a multi-turn conversation.
        history: Prior turns in this conversation (optional).
    """

    question: str
    context: str
    conversation_id: str = "default"
    history: list[ConversationTurn] = []

    @field_validator("question", "context")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Reject blank question or context."""
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v


class AskResponse(BaseModel):
    """Response body for POST /ask.

    Args:
        answer: The numerical answer, or None if the pipeline failed.
        expression: The arithmetic expression that produced the answer.
        conversation_id: Echo of the request conversation_id.
        turn: The turn index for this exchange.
        error: Failure reason if answer is None.
        cached: True if this result was served from the prediction cache.
    """

    answer: float | None
    expression: str | None
    conversation_id: str
    turn: int
    error: str | None = None
    cached: bool = False

    @classmethod
    def from_result(cls, result: PipelineResult, cached: bool = False) -> "AskResponse":
        """Build a response from a PipelineResult."""
        return cls(
            answer=result.answer,
            expression=result.expression,
            conversation_id=result.conversation_id,
            turn=result.turn,
            error=result.error,
            cached=cached,
        )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    provider: str
    model: str
