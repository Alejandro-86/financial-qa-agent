"""Conversation-level data models."""

from pydantic import BaseModel, field_validator


class Question(BaseModel):
    """A single question within a multi-turn conversation.

    Args:
        text: The question text as posed by the user.
        turn: 1-based turn index within the conversation.
        conversation_id: Identifier linking turns in the same conversation.
    """

    text: str
    turn: int
    conversation_id: str

    @field_validator("turn")
    @classmethod
    def turn_must_be_positive(cls, v: int) -> int:
        """Ensure turn index starts at 1."""
        if v < 1:
            raise ValueError("turn must be >= 1")
        return v

    @field_validator("text")
    @classmethod
    def text_cannot_be_empty(cls, v: str) -> str:
        """Reject blank question text."""
        if not v.strip():
            raise ValueError("question text cannot be empty")
        return v


class ConversationTurn(BaseModel):
    """A completed question-answer exchange within a conversation.

    Args:
        question: The original question text.
        answer: The string answer produced by the pipeline, or None if failed.
        numerical_answer: The normalised float answer, or None.
    """

    question: str
    answer: str | None = None
    numerical_answer: float | None = None
