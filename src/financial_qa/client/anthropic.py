"""Anthropic (Claude) client implementation."""

from anthropic import Anthropic

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

    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
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

        response = self._client.messages.create(
            model=kwargs.get("model", self._model),  # type: ignore[arg-type]
            max_tokens=kwargs.get("max_tokens", self._max_tokens),  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", self._temperature),  # type: ignore[arg-type]
            system=system or "",
            messages=chat_messages,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
