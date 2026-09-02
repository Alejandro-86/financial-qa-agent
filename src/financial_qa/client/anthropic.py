"""Anthropic (Claude) client implementation."""

from typing import Any

from anthropic import Anthropic
from anthropic.types import TextBlock

from financial_qa.client.base import LLMClient, LLMMessage, LLMResponse, Role


class AnthropicClient(LLMClient):
    """LLM client backed by the Anthropic Claude API.

    Args:
        model: Claude model identifier (e.g. 'claude-sonnet-4-6').
        api_key: Anthropic API key.
        max_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature (0.0 = deterministic).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Send messages to Claude and return the response.

        System messages are extracted and passed via the Anthropic `system` param.
        All other messages are passed in the `messages` list.
        """
        system = next(
            (m.content for m in messages if m.role == Role.SYSTEM), None
        )
        chat_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]

        model: str = kwargs.get("model", self._model)
        max_tokens: int = kwargs.get("max_tokens", self._max_tokens)

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "",
            messages=chat_messages,  # type: ignore[arg-type]
        )

        text_block = next(b for b in response.content if isinstance(b, TextBlock))
        return LLMResponse(
            content=text_block.text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
