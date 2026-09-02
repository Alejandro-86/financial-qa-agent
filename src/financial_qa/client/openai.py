"""OpenAI client implementation."""

from typing import Any

from openai import OpenAI

from financial_qa.client.base import LLMClient, LLMMessage, LLMResponse


class OpenAIClient(LLMClient):
    """LLM client backed by the OpenAI API.

    Args:
        model: OpenAI model identifier (e.g. 'gpt-4o').
        api_key: OpenAI API key.
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
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Send messages to OpenAI and return the response."""
        oai_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        model: str = kwargs.get("model", self._model)
        max_tokens: int = kwargs.get("max_tokens", self._max_tokens)
        temperature: float = kwargs.get("temperature", self._temperature)

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=oai_messages,  # type: ignore[arg-type]
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
