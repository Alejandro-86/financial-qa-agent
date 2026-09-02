"""Abstract base class and shared types for all LLM provider clients."""

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, field_validator


class Role(StrEnum):
    """Message role in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """A single message in a chat conversation.

    Args:
        role: Who sent this message.
        content: The message text.
    """

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Reject blank message content."""
        if not v.strip():
            raise ValueError("message content cannot be empty")
        return v


class LLMResponse(BaseModel):
    """The response returned by a provider after a completion request.

    Args:
        content: The generated text.
        model: The model identifier used for this completion.
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the completion.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total token consumption for this request."""
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    """Provider-agnostic interface for chat completions.

    Implementations wrap a specific provider SDK (Anthropic, OpenAI, Groq)
    while exposing a uniform interface.  Swap provider via the factory without
    changing any pipeline code.
    """

    @abstractmethod
    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        """Send a list of messages and return the completion.

        Args:
            messages: Ordered conversation history including the new user turn.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).

        Returns:
            LLMResponse containing the generated text and token counts.
        """
